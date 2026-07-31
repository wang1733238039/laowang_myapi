"""
Banana2 ComfyUI Node
支持文生图、图生图、多图生图的异步并发API调用
"""

import asyncio
import json
import random
import time
import requests
import uuid
import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from PIL import Image
import io
import base64

import os as _os

_DEBUG = _os.environ.get("LAOWANG_MYAPI_DEBUG", "0") == "1"

_RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_PATTERNS = (
    "sslerror",
    "ssl eof",
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "connectionerror",
    "connection reset",
    "connection aborted",
    "max retries exceeded",
    "remotedisconnected",
    "brokenpipeerror",
    "connecttimeout",
    "readtimeout",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
)
_NON_RETRYABLE_ERROR_PATTERNS = (
    "invalid api key",
    "unauthorized",
    "forbidden",
    "permission denied",
    "insufficient balance",
    "余额不足",
    "内容违规",
    "content policy",
    "invalid parameter",
    "参数错误",
)


def _dlog(*args, **kwargs):
    """调试日志开关，默认关闭。设置 LAOWANG_MYAPI_DEBUG=1 启用。"""
    if _DEBUG:
        print(*args, **kwargs)


def _is_retryable_error(error_message: str = "", http_status: Optional[int] = None) -> bool:
    """仅将临时网络错误、限流和服务端错误标记为可重试。"""
    if http_status is not None:
        return http_status in _RETRYABLE_HTTP_STATUS

    error_lower = str(error_message).lower()
    if any(pattern in error_lower for pattern in _NON_RETRYABLE_ERROR_PATTERNS):
        return False
    return any(pattern in error_lower for pattern in _RETRYABLE_ERROR_PATTERNS)


def _retry_delay_seconds(failed_attempt_index: int) -> float:
    """指数退避并加入抖动；首次失败等待约 3 秒，最大不超过 30 秒。"""
    base_delay = min(3 * (2 ** failed_attempt_index), 30)
    return base_delay + random.uniform(0, min(base_delay * 0.25, 3))


class Banana2ExecutionError(RuntimeError):
    """所有有效任务均失败时，让错误归属当前 API 节点。"""


# OSS 上传相关导入
try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False
    print("[WARNING] oss2 not available, image URL upload will not work")


def _pil_images_to_oss_urls(
    images: List[Image.Image],
    oss_config: Dict[str, str],
    timeout_seconds: int = 30
) -> List[str]:
    """将PIL图像上传到阿里云OSS，返回图片URL列表"""
    if not OSS_AVAILABLE:
        raise Exception("oss2库不可用，无法上传图片到OSS")

    # 检查OSS配置
    required_keys = ["endpoint", "access_key_id", "access_key_secret", "bucket_name"]
    for key in required_keys:
        if key not in oss_config or not oss_config[key]:
            raise Exception(f"OSS配置缺少必要参数: {key}")

    endpoint = oss_config["endpoint"]
    access_key_id = oss_config["access_key_id"]
    access_key_secret = oss_config["access_key_secret"]
    bucket_name = oss_config["bucket_name"]
    object_prefix = oss_config.get("object_prefix", "uploads/")
    use_signed_url = oss_config.get("use_signed_url", True)
    signed_url_expire_seconds = int(oss_config.get("signed_url_expire_seconds", 3600))
    security_token = oss_config.get("security_token", "")

    # 初始化OSS客户端
    auth = oss2.StsAuth(access_key_id, access_key_secret, security_token) if security_token else oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    urls = []

    for idx, pil_image in enumerate(images):
        try:
            # 将PIL图像转换为PNG字节数据
            bio = io.BytesIO()
            pil_image.save(bio, format="PNG")
            image_bytes = bio.getvalue()

            # 生成对象键
            today = datetime.datetime.utcnow()
            date_path = f"{today.year:04d}/{today.month:02d}/{today.day:02d}"
            uid = uuid.uuid4().hex[:8]
            suggested_name = f"banana_image_{uid}_{idx+1:04d}.png"
            object_key = "/".join(x.strip("/\\") for x in [object_prefix, date_path, suggested_name] if x)
            object_key = object_key.replace("\\", "/")

            # 上传到OSS
            headers = {"Content-Type": "image/png"}
            result = bucket.put_object(object_key, image_bytes, headers=headers)

            if not (200 <= result.status < 300):
                raise Exception(f"OSS上传失败: status={result.status}")

            # 生成URL
            if use_signed_url:
                url = bucket.sign_url("GET", object_key, signed_url_expire_seconds)
            else:
                # 构造公共URL
                scheme = "https"
                ep = endpoint
                if endpoint.startswith("http://"):
                    scheme = "http"
                    ep = endpoint[len("http://"):]
                elif endpoint.startswith("https://"):
                    ep = endpoint[len("https://"):]
                url = f"{scheme}://{bucket_name}.{ep}/{object_key}"

            urls.append(url)
            _dlog(f"[DEBUG] 图片{idx+1}上传成功: {url}")

        except Exception as e:
            print(f"[ERROR] 图片{idx+1}上传失败: {str(e)}")
            raise Exception(f"图片上传失败: {str(e)}")

    return urls


def _calculate_aspect_ratio(width: int, height: int) -> str:
    """计算图片的宽高比，返回最接近的NanoBanana支持的比例"""
    # 验证输入参数
    if width is None or height is None or width <= 0 or height <= 0:
        _dlog(f"[警告] 无效的图片尺寸 width={width}, height={height}，使用默认比例 1:1")
        return "1:1"

    ratio = width / height
    _dlog(f"[调试] 计算宽高比: {width}/{height} = {ratio:.6f}")

    # 定义NanoBanana支持的比例及其阈值
    supported_ratios = {
        "1:1": 1.0,        # 正方形
        "9:16": 9/16,      # 0.5625 (竖屏手机)
        "16:9": 16/9,      # 1.777... (横屏宽屏)
        "3:4": 3/4,        # 0.75 (竖屏)
        "4:3": 4/3,        # 1.333... (横屏)
        "3:2": 3/2,        # 1.5 (横屏)
        "2:3": 2/3,        # 0.666... (竖屏)
        "5:4": 5/4,        # 1.25 (横屏)
        "4:5": 4/5,        # 0.8 (竖屏)
        "21:9": 21/9,      # 2.333... (超宽屏)
    }

    # 找到差值最小的比例
    min_diff = float('inf')
    best_ratio = "1:1"  # 默认值

    for ratio_name, target_ratio in supported_ratios.items():
        diff = abs(ratio - target_ratio)
        _dlog(f"[调试] 比例 {ratio_name} ({target_ratio:.6f}): 差值 = {diff:.6f}")
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio_name

    _dlog(f"[调试] 最终匹配比例: {best_ratio} (差值 = {min_diff:.6f})")
    return best_ratio


def _get_image_size_with_exif(image: Image.Image) -> Tuple[int, int]:
    """获取图片的实际尺寸，考虑EXIF方向信息

    当图片有EXIF方向信息（orientation）时，需要根据方向信息调整宽高。
    例如：如果orientation=6（顺时针旋转90度），则实际显示时需要交换宽高。

    Args:
        image: PIL Image对象

    Returns:
        (width, height): 实际显示的尺寸
    """
    width, height = image.size

    # 检查EXIF方向信息
    try:
        exif = image.getexif()
        orientation = exif.get(274)  # EXIF标签274是Orientation
        if orientation:
            # orientation值说明：
            # 1 = 正常（0度）- 不需要交换
            # 3 = 旋转180度 - 不需要交换（尺寸不变）
            # 6 = 顺时针旋转90度（需要交换宽高）
            # 8 = 逆时针旋转90度（需要交换宽高）
            if orientation in [6, 8]:  # 需要旋转90度或270度
                # 交换宽高
                width, height = height, width
    except Exception:
        # 如果获取EXIF失败或图片没有EXIF信息，使用原始尺寸（已赋值，无需修改）
        pass

    return width, height


class GeminiBatchNode:
    """
    GeminiBatch ComfyUI节点 - 支持并发多组任务处理
    """
    CATEGORY = "laowang_myapi"

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入插槽（按组顺序：image_#.1..image_#.10, prompt_#；prompt为插槽-only）"""
        required = {
            "provider": ("STRING", {
                "default": "comfly",
                "tooltip": "供应商名称"
            }),
            "base_url": ("STRING", {
                "default": "https://ai.comfly.org",
                "tooltip": "API基础地址"
            }),
            "api_key": ("STRING", {
                "tooltip": "API密钥"
            }),
            "model": ("STRING", {
                "default": "nano-banana-2",
                "tooltip": "模型名称 (nano-banana系列)"
            }),
            "mode": (["Text2Img", "Img2Img"], {
                "default": "Img2Img",
                "tooltip": "图像生成模式"
            }),
            "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                "default": "auto",
                "tooltip": "图像宽高比 (auto=根据输入图片自动计算)"
            }),
            "response_format": (["url", "b64_json"], {
                "default": "url",
                "tooltip": "响应格式"
            }),
            "mode": (["Text2Img", "Img2Img"], {
                "default": "Img2Img",
                "tooltip": "图像生成模式"
            }),
            "img_size": (["1K", "2K", "4K"], {
                "default": "2K",
                "tooltip": "图片尺寸"
            }),
            "img_n": ("INT", {
                "default": 1,
                "min": 1,
                "max": 1,
                "tooltip": "生成图片数量 (只能填1)"
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xffffffffffffffff,
                "tooltip": "随机种子值，每次点击重新生成随机符合comfyui规范的种子值"
            }),
            "timeout": ("INT", {
                "default": 200,
                "min": 10,
                "max": 600,
                "tooltip": "每一次请求超时(秒) ，如果超时不管是否返回结果，立即判定超时"
            }),
            "retry_count": ("INT", {
                "default": 0,
                "min": 0,
                "max": 5,
                "tooltip": "每一个请求如果失败后的再次重试次数"
            }),
            "node_enabled": ("BOOLEAN", {
                "default": True,
                "tooltip": "节点开关 若为关程序不执行跳过(视为成功执行)"
            })
        }

        # 可选的组输入（可以为None）
        optional = {
            # OSS配置（用于图片上传）
            "oss_endpoint": ("STRING", {
                "default": "",
                "tooltip": "阿里云OSS endpoint (如: https://oss-cn-hangzhou.aliyuncs.com)"
            }),
            "oss_access_key_id": ("STRING", {
                "default": "",
                "tooltip": "阿里云OSS AccessKey ID"
            }),
            "oss_access_key_secret": ("STRING", {
                "default": "",
                "tooltip": "阿里云OSS AccessKey Secret",
                "password": True
            }),
            "oss_bucket_name": ("STRING", {
                "default": "",
                "tooltip": "阿里云OSS Bucket名称"
            }),
            "oss_object_prefix": ("STRING", {
                "default": "banana-images/",
                "tooltip": "OSS对象前缀路径"
            }),
            "oss_use_signed_url": ("BOOLEAN", {
                "default": True,
                "tooltip": "是否使用签名URL（更安全但有时效性）"
            }),
            "oss_signed_url_expire_seconds": ("INT", {
                "default": 3600,
                "min": 60,
                "max": 604800,
                "tooltip": "签名URL过期时间（秒）"
            }),
            "oss_security_token": ("STRING", {
                "default": "",
                "tooltip": "阿里云STS临时安全令牌（可选）",
                "password": True
            })
        }

        for group in range(1, 11):
            for img_idx in range(1, 11):
                optional[f"image_{group}.{img_idx}"] = ("IMAGE", {
                    "tooltip": f"组{group}的第{img_idx}张参考图像"
                })

            # prompt 仅作为插槽，不在前端显示文本输入框；使用 forceInput=True 强制仅插槽模式
            optional[f"prompt_{group}"] = ("STRING", {
                "tooltip": f"组{group}的文本提示词（仅插槽）",
                "forceInput": True
            })

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",  # 合并输出: images, urls, responses
                   "IMAGE", "STRING", "INT", "STRING",    # group1: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group2: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group3: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group4: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group5: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group6: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group7: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group8: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group9: image, url, response, info
                   "IMAGE", "STRING", "INT", "STRING",    # group10: image, url, response, info
                   "STRING")                              # stats

    RETURN_NAMES = ("images", "urls", "responses",  # 合并输出
                   "group1_image", "group1_url", "group1_response", "group1_info",  # group1
                   "group2_image", "group2_url", "group2_response", "group2_info",  # group2
                   "group3_image", "group3_url", "group3_response", "group3_info",  # group3
                   "group4_image", "group4_url", "group4_response", "group4_info",  # group4
                   "group5_image", "group5_url", "group5_response", "group5_info",  # group5
                   "group6_image", "group6_url", "group6_response", "group6_info",  # group6
                   "group7_image", "group7_url", "group7_response", "group7_info",  # group7
                   "group8_image", "group8_url", "group8_response", "group8_info",  # group8
                   "group9_image", "group9_url", "group9_response", "group9_info",  # group9
                   "group10_image", "group10_url", "group10_response", "group10_info", # group10
                   "stats")  # 统计

    OUTPUT_IS_LIST = (True, False, False,  # images为列表，其他为单个值
                      False, False, False, False,  # group1
                      False, False, False, False,  # group2
                      False, False, False, False,  # group3
                      False, False, False, False,  # group4
                      False, False, False, False,  # group5
                      False, False, False, False,  # group6
                      False, False, False, False,  # group7
                      False, False, False, False,  # group8
                      False, False, False, False,  # group9
                      False, False, False, False,  # group10
                      False)  # stats

    FUNCTION = "execute"
    OUTPUT_NODE = False

    def __init__(self):
        self.session = requests.Session()

    def execute(self, **kwargs):
        """主执行方法"""
        # 检查节点是否启用
        if not kwargs.get("node_enabled", True):
            return self._get_empty_outputs()

        try:
            # ===== 调试信息: 输入参数详情 =====
            _dlog("\n[DEBUG] Banana2Node 执行开始 =====")
            _dlog(f"[INFO] 节点启用状态: {kwargs.get('node_enabled', True)}")
            _dlog(f"[INFO] 基础URL: {kwargs.get('base_url', 'N/A')}")
            _dlog(f"[INFO] API密钥: {'已配置' if kwargs.get('api_key') else '未配置'}")
            _dlog(f"[INFO] 模型: {kwargs.get('model', 'N/A')}")
            _dlog(f"[INFO] 模式: {kwargs.get('mode', 'N/A')}")
            _dlog(f"[INFO] 宽高比: {kwargs.get('aspect_ratio', 'N/A')}")
            _dlog(f"[INFO] 图片尺寸: {kwargs.get('img_size', 'N/A')}")
            _dlog(f"[INFO] 图片数量: {kwargs.get('img_n', 'N/A')}")
            _dlog(f"[INFO] 种子: {kwargs.get('seed', 'N/A')}")
            _dlog(f"[INFO] 响应格式: {kwargs.get('response_format', 'N/A')}")
            _dlog(f"[INFO] 水印: {kwargs.get('watermark', 'N/A')}")
            _dlog(f"[INFO] 流式输出: {kwargs.get('stream', 'N/A')}")
            _dlog(f"[INFO] 并发数: {kwargs.get('concurrency', 'N/A')}")
            _dlog(f"[INFO] 超时时间: {kwargs.get('timeout', 'N/A')}")
            _dlog(f"[INFO] 重试次数: {kwargs.get('retry_count', 'N/A')}")

            # 显示各组的输入状态
            _dlog("\n[DEBUG] 各组输入状态:")
            for group in range(1, 11):
                has_images = any(kwargs.get(f"image_{group}.{i}") is not None for i in range(1, 11))
                prompt = kwargs.get(f"prompt_{group}")
                _dlog(f"  组{group}: 图片={has_images}, 提示词={'有' if prompt else '无'}")

            # 解析输入参数
            config = self._parse_config(kwargs)
            tasks = self._parse_tasks(kwargs, config)

            _dlog(f"\n📊 解析结果: 共{len(tasks)}个任务, 其中{len([t for t in tasks if t['is_valid']])}个有效")
            _dlog("=" * 50)

            # 过滤有效任务
            valid_tasks = [task for task in tasks if task["is_valid"]]

            if not valid_tasks:
                print("Banana2: 没有有效的任务组")
                return self._get_empty_outputs()

            # 执行任务
            try:
                # 首先尝试使用asyncio.run() (推荐方式)
                results = asyncio.run(self._execute_tasks_async(valid_tasks, config))
            except RuntimeError as e:
                # 如果已经有运行中的循环，使用线程执行
                import concurrent.futures
                import threading

                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._execute_tasks_async(valid_tasks, config))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    results = future.result()

            # ===== 调试信息: 执行结果详情 =====
            _dlog("\n[DEBUG] Banana2Node 执行结果汇总:")
            _dlog(f"  [INFO] 总任务数: {len(valid_tasks)}")
            _dlog(f"  [SUCCESS] 成功任务: {len([r for r in results if r.get('success', False)])}")
            _dlog(f"  [ERROR] 失败任务: {len([r for r in results if not r.get('success', False)])}")

            for i, result in enumerate(results, 1):
                status = "[SUCCESS]" if result.get("success", False) else "[ERROR]"
                _dlog(f"  任务{i}: {status} {result.get('info', '无信息')}")

            _dlog("\n[DEBUG] 准备返回最终输出...")
            _dlog("=" * 50)

            if results and all(not result.get("success", False) for result in results):
                failures = []
                for result in results:
                    info = result.get("info", "")
                    try:
                        parsed_info = json.loads(info) if isinstance(info, str) else info
                    except (TypeError, json.JSONDecodeError):
                        parsed_info = {"message": str(info)}
                    failures.append({
                        "group_id": result.get("group_id"),
                        "attempts": result.get("attempts", 1),
                        "message": parsed_info.get("message", str(parsed_info)) if isinstance(parsed_info, dict) else str(parsed_info),
                    })

                raise Banana2ExecutionError(json.dumps({
                    "status": "error",
                    "message": "laowang_GeminiBatch 所有有效任务均失败",
                    "failures": failures,
                }, ensure_ascii=False))

            # 处理结果
            return self._process_results(results)

        except Banana2ExecutionError:
            raise
        except Exception as e:
            print(f"Banana2: 执行出错 - {str(e)}")
            return self._get_empty_outputs()

    def _parse_config(self, kwargs) -> Dict[str, Any]:
        """解析配置参数"""
        config = {
            "provider": kwargs.get("provider", "comfly"),
            "base_url": kwargs.get("base_url", "https://ai.comfly.org"),
            "api_key": kwargs.get("api_key", ""),
            "model": kwargs.get("model", "nano-banana-2"),
            "mode": kwargs.get("mode", "Img2Img"),
            "aspect_ratio": kwargs.get("aspect_ratio", "auto"),
            "response_format": kwargs.get("response_format", "url"),
            "img_size": kwargs.get("img_size", "2K"),
            "img_n": kwargs.get("img_n", 1),
            "seed": kwargs.get("seed", 0),
            "timeout": kwargs.get("timeout", 200),
            "retry_count": kwargs.get("retry_count", 0),
            "node_enabled": kwargs.get("node_enabled", True)
        }

        # 解析OSS配置
        oss_config = {}
        oss_keys = ["endpoint", "access_key_id", "access_key_secret", "bucket_name", "object_prefix", "use_signed_url", "signed_url_expire_seconds", "security_token"]
        for key in oss_keys:
            oss_param_key = f"oss_{key}"
            value = kwargs.get(oss_param_key)
            if value is not None and value != "":
                oss_config[key] = value

        if oss_config:
            config["oss_config"] = oss_config
            _dlog(f"[DEBUG] OSS配置已启用: endpoint={oss_config.get('endpoint', 'N/A')}, bucket={oss_config.get('bucket_name', 'N/A')}")
        else:
            _dlog(f"[DEBUG] OSS配置未启用，使用base64格式")

        # 调试输出配置
        _dlog(f"[DEBUG] 配置解析结果: {config}")
        return config

    def _parse_tasks(self, kwargs, config) -> List[Dict[str, Any]]:
        """解析任务输入"""
        tasks = []
        for group in range(1, 11):
            images = []
            # 在Text2Img模式下，不解析图片输入
            if config["mode"] == "Img2Img":
                for img_idx in range(1, 11):
                    img_key = f"image_{group}.{img_idx}"
                    img = kwargs.get(img_key)
                    if img is not None and not self._is_empty_tensor(img):
                        images.append(self._tensor_to_pil(img))

            prompt = kwargs.get(f"prompt_{group}", "").strip()

            tasks.append({
                "group_id": group,
                "images": images,
                "prompt": prompt,
                "is_valid": self._is_task_valid(images, prompt, config["mode"])
            })

        return tasks

    def _is_empty_tensor(self, tensor: torch.Tensor) -> bool:
        """判断是否为空tensor"""
        if tensor is None:
            return True

        # 检查tensor是否全为0或非常小
        return torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-6)

    def _is_task_valid(self, images: List[Image.Image], prompt: str, mode: str) -> bool:
        """判断任务是否有效"""
        # 执行条件:
        # 文生图模式：该组prompt插槽(prompt_x)为空时候，该组任务不执行API任务（忽略图像输入）
        # 图生图模式：当某一组的十张图像插槽(image_x.1~image_x.10)传入均为空值 或 该组prompt插槽(prompt_x)为空，两个条件满足其中一个时候，该组任务不执行API任务
        # 空值判断：图像为None或空tensor，文本为None或空字符串

        has_valid_images = len(images) > 0
        has_valid_prompt = bool(prompt)

        if mode == "Text2Img":
            # 文生图模式：只有prompt为空时才无效（忽略图像输入）
            return has_valid_prompt
        else:  # Img2Img
            # 图生图模式：图片和prompt都必须有效（同时满足）
            return has_valid_images and has_valid_prompt

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将ComfyUI图像tensor转换为PIL图像"""
        # ComfyUI图像tensor格式: [B, H, W, C], RGB, 0-1范围
        if tensor.dim() == 4:  # 批次维度
            tensor = tensor[0]  # 取第一张

        # 转换为numpy并缩放到0-255
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)

        # 转换为PIL图像
        return Image.fromarray(np_img)

    async def _execute_tasks_async(self, tasks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """异步执行所有任务"""
        # 根据任务数量动态调整并发数，最大5个并发
        max_concurrent = min(len(tasks), 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        print(f"Banana2: 开始执行 {len(tasks)} 个任务，使用 {max_concurrent} 个并发")

        async def execute_single_task(task):
            async with semaphore:
                return await self._execute_single_task_with_retry(task, config)

        # 并发执行所有任务
        results = await asyncio.gather(*[execute_single_task(task) for task in tasks], return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Banana2: 任务{tasks[i]['group_id']}执行异常 - {str(result)}")
                processed_results.append({
                    "group_id": tasks[i]["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,  # 失败
                    "info": json.dumps({
                        "status": "error",
                        "message": f"执行异常: {str(result)}"
                    }, ensure_ascii=False)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task_with_retry(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务（带重试）"""
        retry_count = config["retry_count"]
        total_attempts = retry_count + 1

        for attempt in range(total_attempts):
            session = requests.Session()
            try:
                result = await self._execute_single_task(task, config, session)
                result["attempts"] = attempt + 1
                if result["success"]:
                    return result

                retryable = bool(result.get("retryable", False))
                error_message = self._result_error_message(result)
                if not retryable:
                    print(
                        f"Banana2: 任务{task['group_id']}第{attempt + 1}/{total_attempts}次尝试失败"
                        f"（不可重试）- {error_message}"
                    )
                    return result

                if attempt >= retry_count:
                    print(
                        f"Banana2: 任务{task['group_id']}第{attempt + 1}/{total_attempts}次尝试失败，"
                        f"已用尽重试次数 - {error_message}"
                    )
                    return result

                delay = _retry_delay_seconds(attempt)
                print(
                    f"Banana2: 任务{task['group_id']}第{attempt + 1}/{total_attempts}次尝试失败"
                    f"（临时错误）- {error_message}；{delay:.1f}秒后重试"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                retryable = _is_retryable_error(str(e))
                if retryable and attempt < retry_count:
                    delay = _retry_delay_seconds(attempt)
                    print(
                        f"Banana2: 任务{task['group_id']}第{attempt + 1}/{total_attempts}次尝试异常"
                        f"（临时错误）- {str(e)}；{delay:.1f}秒后重试"
                    )
                    await asyncio.sleep(delay)
                    continue

                print(
                    f"Banana2: 任务{task['group_id']}第{attempt + 1}/{total_attempts}次尝试异常，"
                    f"{'已用尽重试次数' if retryable else '不可重试'} - {str(e)}"
                )
                return {
                    "group_id": task["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "attempts": attempt + 1,
                    "retryable": retryable,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"请求异常: {str(e)}",
                        "error_type": type(e).__name__,
                    }, ensure_ascii=False)
                }
            finally:
                session.close()

        raise RuntimeError("unreachable retry loop")

    @staticmethod
    def _result_error_message(result: Dict[str, Any]) -> str:
        info = result.get("info", "")
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                return info
        if isinstance(info, dict):
            return str(info.get("message", info))
        return str(info)

    async def _execute_single_task(
        self,
        task: Dict[str, Any],
        config: Dict[str, Any],
        session: requests.Session,
    ) -> Dict[str, Any]:
        """执行单个任务"""
        # 构建API请求
        api_url, headers, payload = self._build_api_request(task, config)

        # ===== 调试信息: API请求详情 =====
        _dlog(f"\n[DEBUG] 任务{task['group_id']} API请求构建:")
        _dlog(f"  [URL] 请求URL: {api_url}")
        _dlog(f"  [HEADERS] 请求头: {headers}")
        _dlog(f"  [PAYLOAD] 请求体类型: {type(payload)}")
        _dlog(f"  [IMAGES] 参考图片数量: {len(task['images'])}")
        _dlog(f"  [PROMPT] 提示词: {task['prompt'][:100]}{'...' if len(task['prompt']) > 100 else ''}")
        _dlog("-" * 30)

        is_comfly_provider = config["provider"] == "comfly"

        # 发送请求
        try:
            has_images = len(task["images"]) > 0
            _dlog(f"[DEBUG] 任务{task['group_id']} 开始发送请求, has_images={has_images}, 图片数量={len(task['images'])}")

            # 检查是否使用NanoBanana API（总是使用JSON）
            is_nanobanana_local = config["provider"] in ["BW", "grsai"]

            if is_nanobanana_local:
                # NanoBanana API：总是使用application/json
                _dlog(f"[DEBUG] 任务{task['group_id']} 使用NanoBanana JSON格式发送请求")
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=(15, config["timeout"]),
                    )
                )
            elif has_images:
                # Comfly图生图：multipart/form-data
                _dlog(f"[DEBUG] 任务{task['group_id']} 使用Comfly multipart/form-data发送请求")
                _dlog(f"[DEBUG] data长度: {len(str(payload['data'])) if 'data' in payload else 'N/A'}")
                _dlog(f"[DEBUG] files数量: {len(payload['files']) if 'files' in payload else 'N/A'}")
                request_data = payload["data"]
                files = payload["files"]

                # 如果files中有多个同名文件，需要转换为requests期望的格式
                if isinstance(files.get("image"), list):
                    # 转换为requests期望的列表格式
                    files_list = []
                    for file_tuple in files["image"]:
                        files_list.append(("image", file_tuple))
                    files = files_list

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.post(
                        api_url,
                        headers=headers,
                        data=request_data,
                        files=files,
                        timeout=(15, config["timeout"]),
                    )
                )
            else:
                # Comfly文生图：application/json
                _dlog(f"[DEBUG] 任务{task['group_id']} 使用Comfly JSON格式发送请求")
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=(15, config["timeout"]),
                    )
                )

            _dlog(f"[DEBUG] 任务{task['group_id']} HTTP响应状态码: {response.status_code}")
            _dlog(f"[DEBUG] 响应头: {dict(response.headers)}")

            if response.status_code == 200:
                _dlog(f"[DEBUG] 任务{task['group_id']} 收到200响应，开始解析JSON...")
                _dlog(f"[DEBUG] 响应内容长度: {len(response.text)} 字符")
                _dlog(f"[DEBUG] Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                _dlog(f"[DEBUG] 原始响应文本前300字符: {response.text[:300]}")

                try:
                    result_data = response.json()
                    _dlog(f"[SUCCESS] 任务{task['group_id']} JSON解析成功")
                    _dlog(f"[DEBUG] JSON结构: {type(result_data)}")
                    if isinstance(result_data, dict):
                        _dlog(f"[DEBUG] JSON键: {list(result_data.keys())}")

                    # ===== 调试信息: API响应详情 =====
                    _dlog(f"[SUCCESS] 任务{task['group_id']} API响应成功:")
                    _dlog(f"  [STATUS] 响应状态码: {response.status_code}")
                    _dlog(f"  [RESPONSE] 响应数据: {result_data}")
                    _dlog(f"  [MODE] 异步模式: {config['provider'] in ['BW', 'grsai'] or (config['provider'] == 'comfly' and 'nano-banana' in config['model'])}")
                    _dlog("-" * 30)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] 任务{task['group_id']} JSON解析失败: {str(e)}")
                    print(f"[ERROR] 完整响应文本: {response.text}")
                    return {
                        "group_id": task["group_id"],
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                            "status": "error",
                            "message": f"JSON解析失败: {str(e)}",
                            "response_text": response.text
                        }, ensure_ascii=False)
                    }

                # Comfly供应商和NanoBanana API使用异步模式
                is_nanobanana_local = config["provider"] in ["BW", "grsai"]
                if is_comfly_provider or is_nanobanana_local:
                    return await self._handle_async_response(task["group_id"], result_data, config, session)
                else:
                    # 其他供应商使用同步模式
                    return self._parse_sync_response(task["group_id"], result_data, config["response_format"])
            else:
                print(f"[ERROR] 任务{task['group_id']} API请求失败 - {response.status_code}")
                print(f"[ERROR] 响应内容: {response.text}")
                retryable = _is_retryable_error(http_status=response.status_code)
                return {
                    "group_id": task["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "retryable": retryable,
                    "stage": "submit",
                    "http_status": response.status_code,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"API请求失败 - {response.status_code}",
                        "response_text": response.text
                    }, ensure_ascii=False)
                }

        except requests.exceptions.Timeout as e:
            print(f"Banana2: 任务{task['group_id']} 请求超时 - {str(e)}")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "retryable": True,
                "stage": "submit",
                "info": json.dumps({
                    "status": "error",
                    "message": f"请求超时: {str(e)}",
                    "error_type": type(e).__name__,
                }, ensure_ascii=False)
            }
        except requests.exceptions.RequestException as e:
            print(f"Banana2: 任务{task['group_id']} 请求异常 - {str(e)}")
            retryable = _is_retryable_error(str(e))
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "retryable": retryable,
                "stage": "submit",
                "info": json.dumps({
                    "status": "error",
                    "message": f"请求异常: {str(e)}",
                    "error_type": type(e).__name__,
                }, ensure_ascii=False)
            }
        except Exception as e:
            print(f"Banana2: 任务{task['group_id']} 执行异常 - {str(e)}")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "retryable": False,
                "info": json.dumps({
                    "status": "error",
                    "message": f"执行异常: {str(e)}",
                    "error_type": type(e).__name__,
                }, ensure_ascii=False)
            }

    def _build_api_request(self, task: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, Dict[str, str], Any]:
        """构建API请求"""
        base_url = config["base_url"].rstrip("/")
        has_images = len(task["images"]) > 0
        is_comfly_provider = config["provider"] == "comfly"
        is_nanobanana = config["provider"] in ["BW", "grsai"]

        # 处理aspect_ratio的auto模式
        final_aspect_ratio = config["aspect_ratio"]
        if config["aspect_ratio"] == "auto":
            if has_images:
                # 获取第一张图片的尺寸
                first_image = task["images"][0]
                width, height = _get_image_size_with_exif(first_image)
                if width and height:
                    final_aspect_ratio = _calculate_aspect_ratio(width, height)
                    _dlog(f"[AUTO] 根据输入图片({width}x{height})计算比例: {final_aspect_ratio}")
                else:
                    final_aspect_ratio = "1:1"
                    _dlog("[AUTO] 无法获取图片尺寸，使用默认比例: 1:1")
            else:
                final_aspect_ratio = "1:1"
                _dlog("[AUTO] 无输入图片，使用默认比例: 1:1")

        # 根据mode决定是否使用图像
        use_images = has_images and config["mode"] == "Img2Img"

        # NanoBanana API (仅用于BW供应商，不包括grsai)
        if config["provider"] in ["BW"]:
            # 检查是否使用Pro版本API
            is_pro_model = config["model"] == "nano-banana-pro"

            if is_pro_model:
                # Pro版本API
                api_url = f"{base_url}/api/v1/nanobanana/generate-pro"
            else:
                # 普通版本API
                api_url = f"{base_url}/api/v1/nanobanana/generate"

            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }

            if is_pro_model:
                # Pro版本API参数格式
                payload = {
                    "prompt": task["prompt"],
                    "resolution": config["img_size"],  # Pro版本使用resolution
                    "aspectRatio": final_aspect_ratio,  # Pro版本使用aspectRatio
                    "callBackUrl": base_url  # 使用base_url作为回调地址
                }

                # 添加图像（Pro版本支持最多8张图片，用于图生图）
                if use_images:
                    # 检查是否需要上传图片到OSS
                    oss_config = config.get("oss_config", {})
                    if oss_config and all(k in oss_config and oss_config[k] for k in ["endpoint", "access_key_id", "access_key_secret", "bucket_name"]):
                        # 使用OSS上传图片获取URL
                        try:
                            image_urls = _pil_images_to_oss_urls(
                                images=task["images"][:8],  # Pro版本支持最多8张图片
                                oss_config=oss_config,
                                timeout_seconds=min(config["timeout"], 60)
                            )
                            payload["imageUrls"] = image_urls
                            print(f"NanoBanana Pro: 通过OSS上传了 {len(image_urls)} 张参考图片")
                        except Exception as e:
                            print(f"NanoBanana Pro: OSS上传失败，尝试使用base64: {e}")
                            # 降级到base64方式
                            image_urls = []
                            for img in task["images"][:4]:  # 降级时限制为4张
                                try:
                                    buffer = io.BytesIO()
                                    img.save(buffer, format="PNG")
                                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                    image_urls.append(f"data:image/png;base64,{img_base64}")
                                except Exception as e2:
                                    print(f"NanoBanana Pro: 图片base64处理失败: {e2}")
                                    continue
                            if image_urls:
                                payload["imageUrls"] = image_urls
                                print(f"NanoBanana Pro: 使用base64格式添加了 {len(image_urls)} 张参考图片")
                    else:
                        # 使用base64格式（兼容旧方式）
                        print(f"NanoBanana Pro: 未配置OSS，使用base64格式")
                        image_urls = []
                        for img in task["images"][:4]:  # 限制为4张图片，避免请求过大
                            try:
                                buffer = io.BytesIO()
                                img.save(buffer, format="PNG")
                                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                image_urls.append(f"data:image/png;base64,{img_base64}")
                            except Exception as e:
                                print(f"NanoBanana Pro: 图片处理失败: {e}")
                                continue

                        if image_urls:
                            payload["imageUrls"] = image_urls
                            print(f"NanoBanana Pro: 添加了 {len(image_urls)} 张参考图片（base64格式）")

            else:
                # 普通版本API参数格式
                payload = {
                    "prompt": task["prompt"],
                    "type": "IMAGETOIAMGE" if use_images else "TEXTTOIAMGE",
                    "numImages": config["img_n"],
                    "callBackUrl": ""  # 可以为空，使用轮询模式
                }

                # 添加图像（如果有的话）
                if use_images:
                    # 将PIL图像转换为Base64 URL
                    image_urls = []
                    for img in task["images"]:
                        try:
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            image_urls.append(f"data:image/png;base64,{img_base64}")
                        except Exception as e:
                            print(f"NanoBanana: 图片处理失败: {e}")
                            continue

                    if image_urls:
                        payload["imageUrls"] = image_urls
                        print(f"NanoBanana: 添加了 {len(image_urls)} 张参考图片")

            return api_url, headers, payload

        # grsai API
        elif config["provider"] == "grsai":
            # grsai NanoBanana API
            api_url = f"{base_url}/v1/draw/nano-banana"

            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }

            # 构建请求参数
            # 将comfly的img_size转换为grsai的imageSize格式
            size_mapping = {"1K": "1K", "2K": "2K", "4K": "4K"}
            grsai_image_size = size_mapping.get(config["img_size"], "1K")

            payload = {
                "model": config["model"],
                "prompt": task["prompt"],
                "aspectRatio": final_aspect_ratio,
                "imageSize": grsai_image_size,
                "webHook": "-1",  # 立即返回id，使用轮询模式
                "shutProgress": False
            }

            # 添加图像（图生图模式）
            if use_images:
                # 将PIL图像转换为Base64 URL
                image_urls = []
                for img in task["images"]:
                    try:
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        image_urls.append(f"data:image/png;base64,{img_base64}")
                    except Exception as e:
                        print(f"grsai: 图片处理失败: {e}")
                        continue

                if image_urls:
                    payload["urls"] = image_urls
                    print(f"grsai: 添加了 {len(image_urls)} 张参考图片")

            return api_url, headers, payload

        # Comfly API (原有逻辑)
        elif use_images:
            # 图生图 - 使用multipart/form-data
            api_url = f"{base_url}/v1/images/edits"
            query_params = ""

            # Comfly供应商使用异步模式
            if is_comfly_provider:
                query_params = "?async=true"
                api_url += query_params

            headers = {
                "Authorization": f"Bearer {config['api_key']}"
            }

            # 构建multipart/form-data
            files = {}
            data = {
                "model": config["model"],
                "prompt": task["prompt"],
                "response_format": config["response_format"],
                "aspect_ratio": final_aspect_ratio,
                "image_size": config["img_size"]
            }

            # 添加图像文件 - 支持多图
            # 存储为列表，稍后在发送时转换为正确的requests格式
            image_files = []
            upload_batch_id = uuid.uuid4().hex
            for i, img in enumerate(task["images"]):
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                filename = (
                    f"banana_{upload_batch_id}_"
                    f"g{task['group_id']:02d}_i{i + 1:02d}.png"
                )
                image_files.append((filename, buffer, "image/png"))

            files["image"] = image_files

            return api_url, headers, {"data": data, "files": files}
        else:
            # 文生图 - 使用application/json
            api_url = f"{base_url}/v1/images/generations"
            query_params = ""

            # Comfly供应商使用异步模式
            if is_comfly_provider:
                query_params = "?async=true"
                api_url += query_params

            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": config["model"],
                "prompt": task["prompt"],
                "response_format": config["response_format"],
                "aspect_ratio": final_aspect_ratio,
                "image_size": config["img_size"]
            }

            return api_url, headers, payload

    async def _handle_async_response(
        self,
        group_id: int,
        response_data: Dict[str, Any],
        config: Dict[str, Any],
        session: requests.Session,
    ) -> Dict[str, Any]:
        """处理异步响应，获取task_id并轮询状态"""
        try:
            # 从响应中获取task_id
            task_id = None

            # NanoBanana API响应格式 (包括grsai)
            if config["provider"] in ["BW"]:
                _dlog(f"[DEBUG] NanoBanana响应数据类型检查:")
                _dlog(f"  response_data类型: {type(response_data)}")
                _dlog(f"  response_data是字典: {isinstance(response_data, dict)}")

                if isinstance(response_data, dict):
                    _dlog(f"  response_data内容: {response_data}")
                    code_value = response_data.get("code")
                    has_data = "data" in response_data
                    _dlog(f"  code值: {code_value} (类型: {type(code_value)})")
                    _dlog(f"  包含data字段: {has_data}")

                    if code_value == 200 and has_data:
                        data_field = response_data["data"]
                        _dlog(f"  data字段类型: {type(data_field)}")
                        _dlog(f"  data字段内容: {data_field}")

                        if isinstance(data_field, dict):
                            task_id = data_field.get("taskId")
                            _dlog(f"  提取的task_id: {task_id}")
                        else:
                            # 处理data字段不是字典的情况
                            task_id = str(data_field) if data_field else None
                            _dlog(f"  data不是字典，转换为字符串task_id: {task_id}")

                        if task_id:
                            provider_name = "NanoBanana" if config["provider"] in ["BW"] else ("grsai" if config["provider"] == "grsai" else "Comfly")
                            print(f"{provider_name}: 任务{group_id} 异步任务已提交，task_id: {task_id}")
                            # 开始轮询查询状态
                            return await self._poll_task_status(group_id, task_id, config, session)
                        else:
                            print(f"NanoBanana: 任务{group_id} 响应中未找到task_id")
                            return {
                                "group_id": group_id,
                                "success": False,
                                "image": None,
                                "url": "",
                                "response_code": 2,
                                "info": json.dumps({
                                    "status": "error",
                                    "message": "响应中未找到task_id",
                                    "response_data": response_data
                                }, ensure_ascii=False)
                            }
                    else:
                        print(f"NanoBanana: 任务{group_id} API响应错误: {response_data}")
                        return {
                            "group_id": group_id,
                            "success": False,
                            "image": None,
                            "url": "",
                            "response_code": 2,
                            "info": json.dumps({
                                "status": "error",
                                "message": f"API响应错误: {response_data.get('msg', '未知错误')}",
                                "response_data": response_data
                            }, ensure_ascii=False)
                        }
                else:
                    print(f"NanoBanana: 任务{group_id} 响应数据不是字典类型: {type(response_data)}")
                    return {
                        "group_id": group_id,
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                            "status": "error",
                            "message": f"响应数据类型错误: {type(response_data)}，期望字典类型",
                            "response_data": str(response_data)
                        }, ensure_ascii=False)
                    }

            # grsai API响应格式
            elif config["provider"] == "grsai":
                _dlog(f"[DEBUG] grsai响应数据类型检查:")
                _dlog(f"  response_data类型: {type(response_data)}")
                _dlog(f"  response_data内容: {response_data}")

                if isinstance(response_data, dict):
                    code_value = response_data.get("code")
                    data_field = response_data.get("data")

                    # grsai成功码是0
                    if code_value == 0 and data_field:
                        if isinstance(data_field, dict):
                            task_id = data_field.get("id")
                            _dlog(f"  grsai提取的task_id: {task_id}")

                            if task_id:
                                print(f"grsai: 任务{group_id} 异步任务已提交，task_id: {task_id}")
                                # 开始轮询查询状态
                                return await self._poll_task_status(group_id, task_id, config, session)
                        else:
                            print(f"grsai: 任务{group_id} data字段不是字典类型: {type(data_field)}")

                    print(f"grsai: 任务{group_id} API响应错误: {response_data}")
                    return {
                        "group_id": group_id,
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                            "status": "error",
                            "message": f"API响应错误: {response_data.get('msg', '未知错误')}",
                            "response_data": response_data
                        }, ensure_ascii=False)
                    }
                else:
                    print(f"grsai: 任务{group_id} 响应数据不是字典类型: {type(response_data)}")
                    return {
                        "group_id": group_id,
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                            "status": "error",
                            "message": f"响应数据类型错误: {type(response_data)}，期望字典类型",
                            "response_data": str(response_data)
                        }, ensure_ascii=False)
                    }

            # Comfly API响应格式
            else:
                if "task_id" in response_data:
                    # 直接在响应根层级
                    task_id = response_data["task_id"]
                elif "data" in response_data and isinstance(response_data["data"], dict) and "task_id" in response_data["data"]:
                    # 在data子对象中
                    task_id = response_data["data"]["task_id"]
                elif "data" in response_data and isinstance(response_data["data"], str):
                    # data字段直接是task_id字符串
                    task_id = response_data["data"]

            if task_id:
                provider_name = "NanoBanana" if config["provider"] in ["BW"] else ("grsai" if config["provider"] == "grsai" else "Comfly")
                print(f"{provider_name}: 任务{group_id} 异步任务已提交，task_id: {task_id}")
                # 开始轮询查询状态
                return await self._poll_task_status(group_id, task_id, config, session)
            else:
                provider_name = "NanoBanana" if config["provider"] in ["BW"] else ("grsai" if config["provider"] == "grsai" else "Comfly")
                print(f"{provider_name}: 任务{group_id} 异步响应中未找到task_id: {response_data}")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"异步响应中未找到task_id",
                        "response_data": response_data
                    }, ensure_ascii=False)
                }

        except Exception as e:
            provider_name = "NanoBanana" if config["provider"] in ["BW"] else ("grsai" if config["provider"] == "grsai" else "Banana2")
            print(f"{provider_name}: 任务{group_id} 处理异步响应异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": f"处理异步响应异常: {str(e)}",
                        "response_data": response_data
                    }, ensure_ascii=False)
            }

    async def _poll_task_status(
        self,
        group_id: int,
        task_id: str,
        config: Dict[str, Any],
        session: requests.Session,
    ) -> Dict[str, Any]:
        """轮询查询任务状态，每5秒查询一次"""
        import time
        base_url = config["base_url"].rstrip("/")
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        timeout = config.get("timeout", 200)  # 获取用户设置的超时时间
        max_polls = min(60, timeout // 5)  # 最多轮询次数，同时考虑超时时间
        poll_count = 0
        start_time = time.time()

        while poll_count < max_polls and (time.time() - start_time) < timeout:
            poll_count += 1

            try:
                # 构建查询URL - NanoBanana API使用不同的查询路径
                if config["provider"] in ["BW"]:
                    query_url = f"{base_url}/api/v1/nanobanana/record-info?taskId={task_id}"
                    query_method = "GET"
                    query_body = None
                elif config["provider"] == "grsai":
                    # grsai使用POST请求
                    query_url = f"{base_url}/v1/draw/result"
                    query_method = "POST"
                    query_body = {"id": task_id}
                else:
                    query_url = f"{base_url}/v1/images/tasks/{task_id}"
                    query_method = "GET"
                    query_body = None

                # 发送查询请求
                if query_method == "POST":
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: session.post(query_url, headers=headers, json=query_body, timeout=(10, 30))
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: session.get(query_url, headers=headers, timeout=(10, 30))
                    )

                if response.status_code == 200:
                    status_data = response.json()

                    if "data" in status_data:
                        task_info = status_data["data"]

                        # NanoBanana API使用不同的状态字段 (不包括grsai)
                        if config["provider"] in ["BW"]:
                            success_flag = task_info.get("successFlag", 0)
                            complete_time = task_info.get("completeTime")
                            error_message = task_info.get("errorMessage")

                            if success_flag == 1 and complete_time is not None:
                                # 任务成功完成
                                print(f"NanoBanana: 任务{group_id} 生成成功")
                                return self._parse_nanobanana_success_response(group_id, task_info, config["response_format"])
                            elif success_flag == 0 and complete_time is None:
                                # 任务进行中
                                print(f"NanoBanana: 任务{group_id} 进行中...")
                                await asyncio.sleep(5)
                                continue
                            else:
                                # 任务失败
                                fail_reason = error_message or f"successFlag={success_flag}"
                                print(f"NanoBanana: 任务{group_id} 生成失败 - {fail_reason}")
                                return {
                                    "group_id": group_id,
                                    "success": False,
                                    "image": None,
                                    "url": "",
                                    "response_code": 2,
                                    "info": json.dumps({
                                        "status": "error",
                                        "message": f"任务失败: {fail_reason}",
                                        "task_info": task_info
                                    }, ensure_ascii=False)
                                }

                        # grsai API使用不同的状态字段
                        elif config["provider"] == "grsai":
                            status = task_info.get("status", "")
                            progress = task_info.get("progress", 0)
                            failure_reason = task_info.get("failure_reason", "")
                            error_msg = task_info.get("error", "")

                            print(f"grsai: 任务{group_id} 状态查询 [{poll_count}] - 状态: {status}, 进度: {progress}%")

                            if status == "succeeded":
                                # 任务成功完成
                                print(f"grsai: 任务{group_id} 生成成功")
                                return self._parse_grsai_success_response(group_id, task_info, config["response_format"])
                            elif status == "failed":
                                # 任务失败
                                fail_reason = failure_reason or error_msg or "未知错误"
                                print(f"grsai: 任务{group_id} 生成失败 - {fail_reason}")
                                return {
                                    "group_id": group_id,
                                    "success": False,
                                    "image": None,
                                    "url": "",
                                    "response_code": 2,
                                    "info": json.dumps({
                                        "status": "error",
                                        "message": f"任务失败: {fail_reason}",
                                        "task_info": task_info
                                    }, ensure_ascii=False)
                                }
                            else:
                                # 任务进行中 (running)
                                print(f"grsai: 任务{group_id} 进行中... (状态: {status})")
                                await asyncio.sleep(5)
                                continue
                        else:
                            # Comfly API使用原有的状态字段
                            status = task_info.get("status", "")
                            progress = task_info.get("progress", "0%")

                            print(f"Banana2: 任务{group_id} 状态查询 [{poll_count}] - 状态: {status}, 进度: {progress}")

                            if status == "SUCCESS":
                                # 任务完成，解析结果
                                return self._parse_async_success_response(group_id, task_info, config["response_format"])

                            elif status == "FAILURE":
                                # 任务失败
                                fail_reason = task_info.get("fail_reason", "未知错误")
                                print(f"Banana2: 任务{group_id} 生成失败 - {fail_reason}")
                                return {
                                    "group_id": group_id,
                                    "success": False,
                                    "image": None,
                                    "url": "",
                                    "response_code": 2,
                                    "info": json.dumps(task_info, ensure_ascii=False)
                                }

                            elif status in ["IN_PROGRESS", "NOT_START", "PENDING"]:
                                # 任务进行中，继续等待
                                await asyncio.sleep(5)  # 等待5秒
                                continue

                            else:
                                print(f"Banana2: 任务{group_id} 未知状态: {status}")
                                await asyncio.sleep(5)
                                continue

                    else:
                        print(f"Banana2: 任务{group_id} 状态查询响应格式错误: {status_data}")
                        await asyncio.sleep(5)
                        continue

                else:
                    print(f"Banana2: 任务{group_id} 状态查询失败 - {response.status_code}: {response.text}")
                    await asyncio.sleep(5)
                    continue

            except Exception as e:
                print(f"Banana2: 任务{group_id} 状态查询异常 - {str(e)}")
                await asyncio.sleep(5)
                continue

        # 超时
        elapsed_time = time.time() - start_time
        print(f"Banana2: 任务{group_id} 查询超时，已等待{elapsed_time:.1f}秒 (设置超时: {timeout}秒)")
        return {
            "group_id": group_id,
            "success": False,
            "image": None,
            "url": "",
            "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步查询超时，已等待{elapsed_time:.1f}秒 (设置超时: {timeout}秒)"
                }, ensure_ascii=False)
        }

    def _parse_async_success_response(self, group_id: int, task_info: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析异步成功的响应"""
        try:
            if "data" in task_info and "data" in task_info["data"]:
                image_data = task_info["data"]["data"][0]

                # 提取URL
                image_url = image_data.get("url", "")
                if not image_url:
                    # 尝试b64_json
                    b64_data = image_data.get("b64_json", "")
                    if b64_data:
                        # 将base64转换为图像URL (这里简化处理，实际需要上传到服务器)
                        image_url = f"data:image/png;base64,{b64_data}"

                if image_url:
                    # 根据URL格式决定是否下载图片
                    if image_url.startswith("data:image"):
                        # base64格式，需要下载转换
                        image = self._download_image(image_url)
                        if image:
                            print(f"Banana2: 任务{group_id} 图像生成成功 (Base64)")
                            # 根据response_format决定URL返回值
                            return_url = "b64_ok" if response_format == "b64_json" else image_url
                            # 简化task_info，避免返回完整的b64_json
                            simplified_task_info = {
                                "task_id": task_info.get("task_id", ""),
                                "status": task_info.get("status", ""),
                                "progress": task_info.get("progress", ""),
                                "submit_time": task_info.get("submit_time", ""),
                                "finish_time": task_info.get("finish_time", "")
                            }
                            return {
                                "group_id": group_id,
                                "success": True,
                                "image": image,
                                "url": return_url,
                                "response_code": 1,
                                "info": json.dumps({
                                    "status": "success",
                                    "message": "图像生成成功",
                                    "format": "base64" if response_format == "b64_json" else "url",
                                    "task_info": simplified_task_info
                                }, ensure_ascii=False)
                            }
                    else:
                        # URL格式，直接返回URL，不下载图片
                        print(f"Banana2: 任务{group_id} 图像生成成功 (URL): {image_url}")
                        # 简化task_info，避免返回完整的b64_json
                        simplified_task_info = {
                            "task_id": task_info.get("task_id", ""),
                            "status": task_info.get("status", ""),
                            "progress": task_info.get("progress", ""),
                            "submit_time": task_info.get("submit_time", ""),
                            "finish_time": task_info.get("finish_time", "")
                        }
                        return {
                            "group_id": group_id,
                            "success": True,
                            "image": None,  # URL格式不下载图片
                            "url": image_url,
                            "response_code": 1,
                            "info": json.dumps({
                                "status": "success",
                                "message": "图像生成成功",
                                "format": "url",
                                "task_info": simplified_task_info
                            }, ensure_ascii=False)
                        }

            print(f"Banana2: 任务{group_id} 异步响应解析失败 - {task_info}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步响应解析失败",
                    "response_data": task_info
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"Banana2: 任务{group_id} 异步响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"异步响应解析异常: {str(e)}",
                    "response_data": task_info
                }, ensure_ascii=False)
            }

    def _parse_nanobanana_success_response(self, group_id: int, task_info: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析NanoBanana API成功响应"""
        try:
            response_data = task_info.get("response")
            if not response_data:
                print(f"NanoBanana: 任务{group_id} 成功但无响应数据")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": "任务成功但无响应数据",
                        "task_info": task_info
                    }, ensure_ascii=False)
                }

            # 解析响应数据（可能是JSON字符串或对象）
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    print(f"NanoBanana: 任务{group_id} 响应数据不是有效JSON")
                    return {
                        "group_id": group_id,
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": json.dumps({
                            "status": "error",
                            "message": "响应数据格式错误",
                            "response_data": response_data
                        }, ensure_ascii=False)
                    }

            # 解析图片结果 - Pro版本格式
            if isinstance(response_data, dict):
                # Pro版本：response是对象，包含resultImageUrl
                image_url = response_data.get("resultImageUrl", "")
                if not image_url:
                    # 尝试originImageUrl
                    image_url = response_data.get("originImageUrl", "")

                if image_url:
                    print(f"NanoBanana Pro: 任务{group_id} 找到结果图片URL: {image_url}")
                    # Pro版本直接返回URL，不需要下载base64
                    return {
                        "group_id": group_id,
                        "success": True,
                        "image": None,  # URL格式不下载图片
                        "url": image_url,
                        "response_code": 1,
                        "info": json.dumps({
                            "status": "success",
                            "message": "图像生成成功",
                            "format": "url",
                            "task_info": {
                                "taskId": task_info.get("taskId"),
                                "completeTime": task_info.get("completeTime")
                            }
                        }, ensure_ascii=False)
                    }

            # 兼容旧格式：列表格式
            elif isinstance(response_data, list) and len(response_data) > 0:
                image_data = response_data[0]

                # 提取URL
                image_url = image_data.get("url", "")
                if not image_url:
                    # 尝试b64_json
                    b64_data = image_data.get("b64_json", "")
                    if b64_data:
                        image_url = f"data:image/png;base64,{b64_data}"

                if image_url:
                    if image_url.startswith("data:image"):
                        # base64格式，需要下载转换
                        image = self._download_image(image_url)
                        if image:
                            print(f"NanoBanana: 任务{group_id} 图像生成成功 (Base64)")
                            return_url = "b64_ok" if response_format == "b64_json" else image_url
                            return {
                                "group_id": group_id,
                                "success": True,
                                "image": image,
                                "url": return_url,
                                "response_code": 1,
                                "info": json.dumps({
                                    "status": "success",
                                    "message": "图像生成成功",
                                    "format": "base64",
                                    "task_info": {
                                        "taskId": task_info.get("taskId"),
                                        "completeTime": task_info.get("completeTime")
                                    }
                                }, ensure_ascii=False)
                            }
                    else:
                        # URL格式，直接返回URL
                        print(f"NanoBanana: 任务{group_id} 图像生成成功 (URL): {image_url}")
                        return {
                            "group_id": group_id,
                            "success": True,
                            "image": None,  # URL格式不下载图片
                            "url": image_url,
                            "response_code": 1,
                            "info": json.dumps({
                                "status": "success",
                                "message": "图像生成成功",
                                "format": "url",
                                "task_info": {
                                    "taskId": task_info.get("taskId"),
                                    "completeTime": task_info.get("completeTime")
                                }
                            }, ensure_ascii=False)
                        }

            print(f"NanoBanana: 任务{group_id} 响应解析失败 - {response_data}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": "响应解析失败",
                    "response_data": response_data
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"NanoBanana: 任务{group_id} 响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"响应解析异常: {str(e)}",
                    "task_info": task_info
                }, ensure_ascii=False)
            }

    def _parse_grsai_success_response(self, group_id: int, task_info: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析grsai API成功响应"""
        try:
            # grsai响应格式: {"id": "xxx", "results": [{"url": "xxx", "content": "xxx"}], "progress": 100, "status": "succeeded", ...}
            results = task_info.get("results", [])

            if not results:
                print(f"grsai: 任务{group_id} 成功但无结果数据")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": json.dumps({
                        "status": "error",
                        "message": "任务成功但无结果数据",
                        "task_info": task_info
                    }, ensure_ascii=False)
                }

            # 提取第一张图片的URL
            result_data = results[0]
            image_url = result_data.get("url", "")

            if image_url:
                print(f"grsai: 任务{group_id} 图像生成成功 (URL): {image_url}")
                return {
                    "group_id": group_id,
                    "success": True,
                    "image": None,  # URL格式不下载图片
                    "url": image_url,
                    "response_code": 1,
                    "info": json.dumps({
                        "status": "success",
                        "message": "图像生成成功",
                        "format": "url",
                        "task_info": {
                            "id": task_info.get("id"),
                            "status": task_info.get("status"),
                            "progress": task_info.get("progress")
                        }
                    }, ensure_ascii=False)
                }

            print(f"grsai: 任务{group_id} 结果中无URL - {task_info}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": "结果中无URL",
                    "task_info": task_info
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"grsai: 任务{group_id} 响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"响应解析异常: {str(e)}",
                    "task_info": task_info
                }, ensure_ascii=False)
            }

    def _parse_sync_response(self, group_id: int, response_data: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析同步响应"""
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                image_data = response_data["data"][0]

                # 提取URL
                image_url = image_data.get("url", "")
                if not image_url:
                    # 尝试b64_json
                    b64_data = image_data.get("b64_json", "")
                    if b64_data:
                        # 将base64转换为图像URL (这里简化处理，实际需要上传到服务器)
                        image_url = f"data:image/png;base64,{b64_data}"

                if image_url:
                    # 根据response_format决定是否下载图片
                    # 如果是URL格式，直接返回URL；如果是base64，需要下载转换
                    if image_url.startswith("data:image"):
                        # base64格式，需要下载转换
                        image = self._download_image(image_url)
                        if image:
                            # 根据response_format决定URL返回值
                            return_url = "b64_ok" if response_format == "b64_json" else image_url
                            # 简化task_info，避免返回完整的b64_json
                            simplified_task_info = {
                                "task_id": response_data.get("task_id", ""),
                                "status": "SUCCESS",
                                "format": response_format
                            }
                            return {
                                "group_id": group_id,
                                "success": True,
                                "image": image,
                                "url": return_url,
                                "response_code": 1,
                                "info": json.dumps({
                                    "status": "success",
                                    "message": "图像生成成功",
                                    "format": "base64" if response_format == "b64_json" else "url",
                                    "task_info": simplified_task_info
                                }, ensure_ascii=False)
                            }
                    else:
                        # URL格式，直接返回URL，不下载图片
                        _dlog(f"[INFO] URL格式响应，直接返回链接: {image_url}")
                        # 简化task_info，避免返回完整的b64_json
                        simplified_task_info = {
                            "task_id": response_data.get("task_id", ""),
                            "status": "SUCCESS",
                            "format": response_format
                        }
                        return {
                            "group_id": group_id,
                            "success": True,
                            "image": None,  # URL格式不下载图片
                            "url": image_url,
                            "response_code": 1,
                            "info": json.dumps({
                                "status": "success",
                                "message": "图像生成成功",
                                "format": "url",
                                "task_info": simplified_task_info
                            }, ensure_ascii=False)
                        }

            print(f"Banana2: 任务{group_id} 响应解析失败 - {response_data}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"同步响应解析失败",
                    "response_data": response_data
                }, ensure_ascii=False)
            }

        except Exception as e:
            print(f"Banana2: 任务{group_id} 响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": json.dumps({
                    "status": "error",
                    "message": f"同步响应解析异常: {str(e)}",
                    "response_data": response_data
                }, ensure_ascii=False)
            }

    def _download_image(self, url: str, max_retries: int = 2) -> Optional[Image.Image]:
        """下载图像，带重试和完整性检查"""
        for attempt in range(max_retries + 1):
            try:
                if url.startswith("data:image"):
                    # base64数据 - 简化处理，不使用verify()
                    header, data = url.split(",", 1)
                    img_data = base64.b64decode(data)
                    img_buffer = io.BytesIO(img_data)
                    img = Image.open(img_buffer)
                    img.load()  # 确保图像数据被完全加载

                    # 对于base64数据，如果能成功打开图片，说明数据完整
                    # 不需要额外的verify()验证（verify()会关闭图片对象）
                    _dlog(f"[SUCCESS] Base64图片处理成功")
                    return img

                else:
                    # URL下载
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        img_buffer = io.BytesIO(response.content)
                        img = Image.open(img_buffer)

                        # 验证图片完整性
                        img.verify()  # 检查图片是否完整
                        img.close()
                        img_buffer.seek(0)  # 重置buffer位置
                        img = Image.open(img_buffer)  # 重新打开

                        _dlog(f"[SUCCESS] URL图片下载并验证成功，大小: {len(response.content)} bytes")
                        return img
                    else:
                        print(f"[ERROR] 图片下载失败，状态码: {response.status_code}")

            except Exception as e:
                print(f"[ERROR] 图片处理失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    import time
                    time.sleep(1)  # 等待1秒后重试
                    continue

        print(f"[ERROR] 图片下载失败，已重试 {max_retries + 1} 次")
        return None

    def _process_results(self, results: List[Dict[str, Any]]) -> Tuple:
        """处理结果并返回输出"""
        # 分组结果
        group_results = {}
        for result in results:
            group_results[result["group_id"]] = result

        # 统计信息
        valid_tasks = len(results)
        success_tasks = sum(1 for r in results if r["success"])

        # 合并输出：只包含成功的结果
        successful_images = []
        all_urls = []
        all_responses = []

        # 独立组输出
        group_outputs = []

        for group_id in range(1, 11):
            if group_id in group_results:
                result = group_results[group_id]
                # 合并输出
                all_urls.append(result["url"] if result["url"] else "")
                all_responses.append(result["response_code"])

                # 独立组输出
                if result["success"]:
                    if result["image"]:
                        # 有图片数据（Base64格式），需要转换
                        tensor_image = self._pil_to_tensor(result["image"])
                        if tensor_image is not None:
                            # 为group输出添加批次维度 [1, H, W, C]
                            group_image = tensor_image.unsqueeze(0)
                            # 将批次格式的图像加入成功列表（保持每个元素为 [1, H, W, C]）
                            successful_images.append(group_image)
                            group_outputs.extend([
                                group_image,
                                result["url"],
                                result["response_code"],
                                result.get("info", "成功")
                            ])
                        else:
                            # 图片转换失败，当作失败处理
                            print(f"[ERROR] 任务{group_id} 图片转换失败，使用空图片")
                            empty_image = torch.zeros((1, 64, 64, 3))  # 批次格式: [B, H, W, C]
                            group_outputs.extend([
                                empty_image,
                                result["url"],
                                3,  # 转换失败
                                "图片转换失败"
                            ])
                    else:
                        # 无图片数据但有URL（URL格式），创建占位符图像
                        _dlog(f"[INFO] 任务{group_id} URL格式响应: {result['url']}")
                        # 为URL格式创建一个特殊的占位符图像，表示这是URL链接
                        url_placeholder = torch.full((1, 64, 64, 3), 0.5)  # 批次格式，灰色占位符，0-1范围
                        group_outputs.extend([
                            url_placeholder,
                            result["url"],
                            result["response_code"],
                            result.get("info", "URL格式响应")
                        ])
                else:
                    # 失败情况下的独立组输出
                    # ComfyUI图像格式: [B, H, W, C]，torch.Tensor
                    empty_image = torch.zeros((1, 64, 64, 3))
                    group_outputs.extend([
                        empty_image,
                        result["url"],
                        result["response_code"],
                        result.get("info", "未执行")
                    ])
            else:
                # 未执行的任务
                all_urls.append("")
                all_responses.append(0)
                # ComfyUI图像格式: [B, H, W, C]
                empty_image = torch.zeros((1, 64, 64, 3))
                group_outputs.extend([
                    empty_image,
                    "",
                    0,
                    "未执行的任务"
                ])

        # 合并输出images：返回图像列表
        if successful_images:
            merged_images = successful_images  # 直接返回列表，元素为 [1, H, W, C]
            _dlog(f"[DEBUG] 合并图像列表长度: {len(merged_images)}")
        else:
            # 如果没有成功的图像，返回包含单个空图像（批次格式）的列表
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            merged_images = [empty_image]  # 包含单个空图像的列表（批次格式）
            _dlog(f"[DEBUG] 空合并图像列表 (占位符)")

        # urls和responses作为JSON字符串
        urls_json = json.dumps(all_urls, ensure_ascii=False)
        responses_json = json.dumps(all_responses, ensure_ascii=False)

        # 统计输出
        stats = f"(有效任务:{valid_tasks}, 成功任务:{success_tasks})"

        # 返回所有输出：合并输出(3) + 独立组输出(30) + 统计输出(1) = 34个
        return tuple([merged_images, urls_json, responses_json] + group_outputs + [stats])

    def _mask_b64_json(self, data: Any) -> Any:
        """屏蔽API响应中的b64_json内容以避免日志溢出"""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key == "b64_json" and isinstance(value, str) and len(value) > 20:
                    # 只保留前20个字符，并显示数据长度
                    data_length = len(value)
                    masked[key] = f"{value[:20]}...[BASE64_DATA_{data_length}_CHARS]"
                else:
                    masked[key] = self._mask_b64_json(value)
            return masked
        elif isinstance(data, list):
            return [self._mask_b64_json(item) for item in data]
        else:
            return data

    def _pil_to_tensor(self, image: Image.Image) -> Optional[torch.Tensor]:
        """将PIL图像转换为ComfyUI期望的torch.Tensor格式，带错误检查"""
        try:
            if image is None:
                print("[ERROR] 输入图像为空")
                return None

            # 注意：图片已经在_download_image中验证过了，这里不再重复验证
            # 如果图片能到达这里，说明它已经是有效的PIL图像

            # 确保RGB模式
            if image.mode != "RGB":
                _dlog(f"[INFO] 转换图片模式: {image.mode} -> RGB")
                image = image.convert("RGB")

            # 检查图片尺寸
            width, height = image.size
            if width == 0 or height == 0:
                print(f"[ERROR] 图片尺寸无效: {width}x{height}")
                return None

            # 转换为numpy数组，保持0-255范围
            _dlog(f"[INFO] 转换图片尺寸: {width}x{height}")
            np_img = np.array(image)

            # 检查数组形状
            if len(np_img.shape) != 3 or np_img.shape[2] != 3:
                print(f"[ERROR] 图片数组形状异常: {np_img.shape}")
                return None

            # 转换为torch.Tensor，归一化到0-1范围，格式: [H, W, C] (ComfyUI标准格式)
            tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)

            _dlog(f"[SUCCESS] 图片转换为torch.Tensor成功，形状: {tensor.shape}")
            return tensor

        except Exception as e:
            print(f"[ERROR] 图片转tensor失败: {str(e)}")
            return None

    def _get_empty_outputs(self) -> Tuple:
        """返回空的输出"""
        # ComfyUI图像格式: torch.Tensor [B, H, W, C]，范围0-1（批次格式）
        empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 合并输出 - images现在是列表格式（每项为批次格式tensor），返回包含单个空图像的列表
        merged_outputs = [[empty_image], "[]", "[]"]

        # 独立组输出 (10组 × 4)，使用批次格式占位符
        group_outputs = []
        for _ in range(10):
            group_outputs.extend([empty_image, "", 0, "未执行的任务"])

        # 统计输出
        stats_output = ["(有效任务:0, 成功任务:0)"]

        return tuple(merged_outputs + group_outputs + stats_output)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "laowang_GeminiBatch": GeminiBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "laowang_GeminiBatch": "laowang_GeminiBatch"
}
