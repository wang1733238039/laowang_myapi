"""
ModelCompareNode ComfyUI Node
单组图多模型比较节点，同时运行多个模型进行对比
"""

import asyncio
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from PIL import Image
import io
import base64

import os as _os

_DEBUG = _os.environ.get("LAOWANG_MYAPI_DEBUG", "0") == "1"


def _dlog(*args, **kwargs):
    """调试日志开关，默认关闭。设置 LAOWANG_MYAPI_DEBUG=1 启用。"""
    if _DEBUG:
        print(*args, **kwargs)



def _calculate_aspect_ratio(width: int, height: int) -> str:
    """计算图片的宽高比，返回最接近的预设比例"""
    if width is None or height is None or width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    supported_ratios = {
        "1:1": 1.0,
        "9:16": 9/16,
        "16:9": 16/9,
        "3:4": 3/4,
        "4:3": 4/3,
        "3:2": 3/2,
        "2:3": 2/3,
        "5:4": 5/4,
        "4:5": 4/5,
        "21:9": 21/9,
    }
    min_diff = float('inf')
    best = "1:1"
    for name, tgt in supported_ratios.items():
        diff = abs(ratio - tgt)
        if diff < min_diff:
            min_diff = diff
            best = name
    return best


def _get_image_size_with_exif(image: Image.Image) -> Tuple[int, int]:
    width, height = image.size
    try:
        exif = image.getexif()
        orientation = exif.get(274)
        if orientation in [6, 8]:
            width, height = height, width
    except Exception:
        pass
    return width, height

class ModelCompareNode:
    """
    ModelCompareNode ComfyUI节点 - 单组输入，多模型并发比较
    """
    CATEGORY = "laowang_myapi"

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入插槽"""
        return {
            "required": {
            # 单组输入：prompt + 模式；参考图像为可选插槽（最多4张参考图）
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "文本提示词"
                }),
                # image_1 为首要参考图（无需额外索引）
                "mode": (["Text2Img", "Img2Img"], {
                    "default": "Img2Img",
                    "tooltip": "图像生成模式"
                }),
                # 模型1配置 (banana)
                "provider1": ("STRING", {
                    "default": "comfly",
                    "tooltip": "供应商名称"
                }),
                "base_url1": ("STRING", {
                    "default": "https://ai.comfly.chat",
                    "tooltip": "API基础地址"
                }),
                "api_key1": ("STRING", {
                    "tooltip": "API密钥"
                }),
                "model1": ("STRING", {
                    "default": "nano-banana-2",
                    "tooltip": "模型名称 (nano-banana系列)"
                }),
                "aspect_ratio1": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "auto",
                    "tooltip": "图像宽高比 (auto=根据输入图片自动计算)"
                }),
                "response_format1": (["url", "b64_json"], {
                    "default": "url",
                    "tooltip": "响应格式"
                }),
                "img_size1": (["1K", "2K", "4K"], {
                    "default": "2K",
                    "tooltip": "图片尺寸"
                }),
                "img_num1": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1,
                    "tooltip": "生成图片数量 (只能填1)"
                }),
                "api_enabled1": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "模型1的开关，如果为关，该模型不做请求"
                }),
                # 模型2配置 (豆包)
                "provider2": ("STRING", {
                    "default": "comfly",
                    "tooltip": "供应商名称"
                }),
                "base_url2": ("STRING", {
                    "default": "https://ai.comfly.chat",
                    "tooltip": "API基础地址"
                }),
                "api_key2": ("STRING", {
                    "tooltip": "API密钥"
                }),
                "model2": ("STRING", {
                    "default": "doubao-seedream-4-5-251128",
                    "tooltip": "豆包模型"
                }),
                "aspect_ratio2": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "auto",
                    "tooltip": "图像宽高比 (auto=根据输入图片自动计算)"
                }),
                "response_format2": (["url", "b64_json"], {
                    "default": "url",
                    "tooltip": "响应格式"
                }),
                "img_size2": (["2K", "4K"], {
                    "default": "2K",
                    "tooltip": "图片尺寸(2K\\4K)"
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "生成图片数量 (1-4)"
                }),
                "watermark2": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
                "stream2": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否流式响应"
                }),
                "api_enabled2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "模型2的开关，如果为关，该模型不做请求"
                }),
                # 通用参数
                "timeout": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 600,
                    "tooltip": "每一次请求超时(秒) ，如果超时不管是否返回结果，立即判定超时"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子值，每次点击重新生成随机符合comfyui规范的种子值"
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
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "参考图像1"}),
                "image_2": ("IMAGE", {"tooltip": "参考图像2"}),
                "image_3": ("IMAGE", {"tooltip": "参考图像3"}),
                "image_4": ("IMAGE", {"tooltip": "参考图像4"}),
                "image_5": ("IMAGE", {"tooltip": "参考图像5"}),
                "image_6": ("IMAGE", {"tooltip": "参考图像6"}),
                "image_7": ("IMAGE", {"tooltip": "参考图像7"}),
                "image_8": ("IMAGE", {"tooltip": "参考图像8"}),
                "image_9": ("IMAGE", {"tooltip": "参考图像9"}),
                "image_10": ("IMAGE", {"tooltip": "参考图像10"}),
                # OSS配置（用于NanoBanana模型）
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
        }

    # 返回：合并输出(images, urls, responses) + 模型1(4) + 模型2(4) + 比较统计(1)
    RETURN_TYPES = (
        "IMAGE", "STRING", "STRING",  # merged: images(list), urls(json), responses(json)
        "IMAGE", "STRING", "INT", "STRING",  # model1: image, url, response, info
        "IMAGE", "STRING", "INT", "STRING",  # model2: image, url, response, info
        "STRING"  # comparison_stats
    )

    RETURN_NAMES = (
        "images", "urls", "responses",
        "model1_image", "model1_url", "model1_response", "model1_info",
        "model2_image", "model2_url", "model2_response", "model2_info",
        "comparison_stats"
    )

    FUNCTION = "execute"
    OUTPUT_NODE = False

    # 标记第一个输出为列表（images 返回 List[IMAGE]），其余为单值
    OUTPUT_IS_LIST = (
        True,  # images list
        False, False,  # urls, responses
        False, False, False, False,  # model1 outputs
        False, False, False, False,  # model2 outputs
        False  # comparison_stats
    )

    def __init__(self):
        self.session = requests.Session()

    def execute(self, **kwargs):
        """主执行方法"""
        # 检查节点是否启用
        if not kwargs.get("node_enabled", True):
            return self._get_empty_outputs()

        try:
            # ===== 调试信息: 输入参数详情 =====
            _dlog("\n[DEBUG] ModelCompareNode 执行开始 =====")
            _dlog(f"[INFO] 节点启用状态: {kwargs.get('node_enabled', True)}")

            # 显示两个模型的配置
            for i in [1, 2]:
                _dlog(f"[MODEL] 模型{i}配置:")
                _dlog(f"  [URL] 基础URL{i}: {kwargs.get(f'base_url{i}', 'N/A')}")
                _dlog(f"  [KEY] API密钥{i}: {'已配置' if kwargs.get(f'api_key{i}') else '未配置'}")
                _dlog(f"  [MODEL] 模型{i}: {kwargs.get(f'model{i}', 'N/A')}")
                _dlog(f"  [SIZE] 图片尺寸{i}: {kwargs.get(f'img_size{i}', 'N/A')}")
                _dlog(f"  [COUNT] 图片数量{i}: {kwargs.get(f'img_num{i}', 'N/A')}")

            # 显示输入状态
            has_images = any(kwargs.get(f"image{i}") is not None for i in range(1, 11))
            prompt = kwargs.get("prompt")
            _dlog("\n[DEBUG] 输入状态:")
            _dlog(f"  [IMAGES] 参考图片: {'有' if has_images else '无'}")
            _dlog(f"  [PROMPT] 提示词: {'有' if prompt else '无'}")
            _dlog(f"  [PROMPT] 提示词内容: {prompt[:100] if prompt else 'N/A'}{('...' if prompt and len(prompt) > 100 else '')}")

            # 解析输入
            task = self._parse_single_task(kwargs)

            # 默认以 image_1（在 task["images"] 的首位）作为首要参考图

            if not self._is_task_valid(task["images"], task["prompt"], kwargs.get("mode", "Img2Img")):
                print("ModelCompare: 没有有效的输入")
                return self._get_empty_outputs()

            # 并发执行两个模型
            try:
                # 首先尝试使用asyncio.run() (推荐方式)
                results = asyncio.run(self._execute_model_comparison(task, kwargs))
            except RuntimeError as e:
                # 如果已经有运行中的循环，使用线程执行
                import concurrent.futures
                import threading

                def run_async():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._execute_model_comparison(task, kwargs))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    results = future.result()

            # ===== 调试信息: 最终输出详情 =====
            _dlog("\n[DEBUG] ModelCompareNode 准备返回最终输出...")
            _dlog("=" * 50)

            # 处理结果
            return self._process_comparison_results(results)

        except Exception as e:
            print(f"ModelCompare: 执行出错 - {str(e)}")
            return self._get_empty_outputs()

    def _parse_single_task(self, kwargs) -> Dict[str, Any]:
        """解析单组任务输入"""
        images = []
        # 在Text2Img模式下，不解析图片输入
        if kwargs.get("mode", "Img2Img") == "Img2Img":
            for i in range(1, 11):
                img_key = f"image_{i}"
                img = kwargs.get(img_key)
                if img is not None and not self._is_empty_tensor(img):
                    images.append(self._tensor_to_pil(img))

        prompt = kwargs.get("prompt", "").strip()

        return {
            "images": images,
            "prompt": prompt
        }

    def _is_empty_tensor(self, tensor: torch.Tensor) -> bool:
        """判断是否为空tensor"""
        if tensor is None:
            return True
        return torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-6)

    def _is_task_valid(self, images: List[Image.Image], prompt: str, mode: str) -> bool:
        """判断任务是否有效"""
        # 执行条件:
        # 文生图模式：该组prompt插槽(prompt_x)为空时候，该组任务不执行API任务
        # 图生图模式：当某一组的四个图像插槽(image_x.1~image_x.4)传入均为空值 或 该组prompt插槽(prompt_x)为空，两个条件满足其中一个时候，该组任务不执行API任务
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
        if tensor.dim() == 4:  # 批次维度
            tensor = tensor[0]  # 取第一张

        # 转换为numpy并缩放到0-255
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img)

    async def _execute_model_comparison(self, task: Dict[str, Any], kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """并发执行两个模型的比较"""
        print("ModelCompare: 开始执行模型比较 (banana vs Doubao)")

        # 构建两个模型的配置
        banana_config = {
            "provider": kwargs.get("provider1", "comfly"),
            "base_url": kwargs.get("base_url1", "https://ai.comfly.chat"),
            "api_key": kwargs.get("api_key1", ""),
            "model": kwargs.get("model1", "nano-banana-2"),
            "aspect_ratio": kwargs.get("aspect_ratio1", "auto"),
            "response_format": kwargs.get("response_format1", "url"),
            "img_size": kwargs.get("img_size1", "2K"),
            "img_n": kwargs.get("img_num1", 1),
            "mode": kwargs.get("mode", "Img2Img"),
            "seed": kwargs.get("seed", 0),
            "timeout": kwargs.get("timeout", 200),
            "retry_count": kwargs.get("retry_count", 0),
            "api_enabled": kwargs.get("api_enabled1", True)
        }

        doubao_config = {
            "provider": kwargs.get("provider2", "comfly"),
            "base_url": kwargs.get("base_url2", "https://ai.comfly.chat"),
            "api_key": kwargs.get("api_key2", ""),
            "model": kwargs.get("model2", "doubao-seedream-4-5-251128"),
            "aspect_ratio": kwargs.get("aspect_ratio2", "auto"),
            "response_format": kwargs.get("response_format2", "url"),
            "img_size": kwargs.get("img_size2", "2K"),
            "n": kwargs.get("n", 1),
            "mode": kwargs.get("mode", "Img2Img"),
            "seed": kwargs.get("seed", 0),
            "watermark": kwargs.get("watermark2", False),
            "stream": kwargs.get("stream2", False),
            "timeout": kwargs.get("timeout", 200),
            "retry_count": kwargs.get("retry_count", 0),
            "api_enabled": kwargs.get("api_enabled2", True)
        }

        # ===== 调试信息: 模型配置详情 =====
        _dlog("\n[DEBUG] 模型配置详情:")
        _dlog(f"  [banana] banana配置: {banana_config}")
        _dlog(f"  [DOUBAO] Doubao配置: {doubao_config}")
        _dlog(f"  [TASK] 任务详情: 图片数={len(task['images'])}, 提示词长度={len(task['prompt'])}")
        _dlog("-" * 30)

        # 并发执行两个模型
        _dlog("[INFO] 开始并发执行两个模型...")
        results = await asyncio.gather(
            self._execute_banana_model(task, banana_config, kwargs),
            self._execute_doubao_model(task, doubao_config),
            return_exceptions=True
        )

        # ===== 调试信息: 执行结果详情 =====
        _dlog("\n[DEBUG] 模型执行结果:")
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                _dlog(f"  [ERROR] 模型{i} 执行异常: {str(result)}")
            else:
                status = "[SUCCESS]" if result.get("success", False) else "[ERROR]"
                _dlog(f"  {status} 模型{i} ({'banana' if i==1 else 'Doubao'}): {result.get('info', '无信息')}")

        _dlog("-" * 30)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            model_name = "banana" if i == 0 else "Doubao"
            if isinstance(result, Exception):
                print(f"ModelCompare: {model_name}模型执行异常 - {str(result)}")
                processed_results.append({
                    "model": model_name,
                    "success": False,
                    "image": None,
                    "url": "",
                    "status": 2
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_banana_model(self, task: Dict[str, Any], config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型1（支持Comfly和NanoBanana）"""
        provider = config.get("provider", "comfly")
        model = config.get("model", "N/A")
        print(f"[{provider.upper()}] 执行{provider}模型: {model}")

        # 检查模型是否启用
        if not config.get("api_enabled", True):
            print(f"[{provider.upper()}] {provider}模型已被禁用")
            return {
                "model": provider.capitalize(),
                "success": False,
                "image": None,
                "url": "",
                "status": 0
            }

        # 根据provider选择处理逻辑
        # BW和grsai供应商使用NanoBanana API，comfly供应商使用Comfly API
        is_nanobanana_provider = provider in ["BW", "grsai"]

        if is_nanobanana_provider:
            # NanoBanana API需要特殊处理（OSS上传等）
            return await self._execute_nanobanana_model(task, config, kwargs)
        else:
            # banana API使用原逻辑
            return await self._execute_comfly_only_model(task, config)

    async def _execute_comfly_only_model(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行Comfly模型（不包括NanoBanana）"""
        provider = config.get("provider", "comfly")

        # 动态导入 banana2_batch_node，兼容不同运行上下文（直接模块或作为包导入）
        try:
            import importlib
            GeminiBatchNode = importlib.import_module("banana2_batch_node").GeminiBatchNode
        except Exception:
            import importlib.util
            import sys
            import os
            mod_path = os.path.join(os.path.dirname(__file__), "banana2_batch_node.py")
            spec = importlib.util.spec_from_file_location("banana2_batch_node", mod_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["banana2_batch_node"] = module
            spec.loader.exec_module(module)
            GeminiBatchNode = module.GeminiBatchNode

        # 创建虚拟的单组任务
        virtual_kwargs = {
            "prompt_1": task["prompt"],
            "provider": config["provider"],
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "model": config["model"],
            "mode": config["mode"],
            "aspect_ratio": config["aspect_ratio"],
            "response_format": config["response_format"],
            "img_size": config["img_size"],
            "img_n": config["img_n"],
            "timeout": config["timeout"],
            "retry_count": config["retry_count"],
            "node_enabled": True
        }

        # 调试：打印 aspect_ratio 流向与图片信息，帮助排查比例计算问题
        try:
            _dlog(f"[DEBUG] Comfly virtual aspect_ratio (config): {config.get('aspect_ratio')}")
            _dlog(f"[DEBUG] Comfly virtual aspect_ratio (virtual_kwargs): {virtual_kwargs.get('aspect_ratio')}")
            if task.get("images"):
                for i, img in enumerate(task["images"], 1):
                    try:
                        _dlog(f"[DEBUG] Comfly task image{i} size: {img.size} mode: {img.mode}")
                    except Exception:
                        _dlog(f"[DEBUG] Comfly task image{i} type: {type(img)}")
            else:
                _dlog("[DEBUG] Comfly task has no images")
        except Exception:
            pass

        # 保持传入的 aspect_ratio（例如 "auto" 或指定比例），由子节点自行处理

        # 在Img2Img模式下才传递图片（最多传 10 张参考图到 image_1.1..image_1.10）
        if config["mode"] == "Img2Img" and task["images"]:
            for idx, img in enumerate(task["images"][:10], start=1):
                virtual_kwargs[f"image_1.{idx}"] = self._pil_to_tensor(img)

        # 使用GeminiBatchNode的逻辑执行
        banana_node = GeminiBatchNode()
        tasks_parsed = banana_node._parse_tasks(virtual_kwargs, config)
        if tasks_parsed and tasks_parsed[0]["is_valid"]:
            result = await banana_node._execute_single_task_with_retry(tasks_parsed[0], config)
            return {
                "model": provider.capitalize(),
                "success": result["success"],
                    "image": result.get("image"),
                    "url": result.get("url", ""),
                    "status": result.get("response_code", 0),
                    "info": result.get("info", "")
            }

        return {
            "model": provider.capitalize(),
            "success": False,
            "image": None,
            "url": "",
            "status": 0
        }

    async def _execute_nanobanana_model(self, task: Dict[str, Any], config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """执行NanoBanana模型（需要OSS上传）"""
        provider = config.get("provider", "BW")

        # 动态导入 banana2_batch_node
        try:
            import importlib
            GeminiBatchNode = importlib.import_module("banana2_batch_node").GeminiBatchNode
        except Exception:
            import importlib.util
            import sys
            import os
            mod_path = os.path.join(os.path.dirname(__file__), "banana2_batch_node.py")
            spec = importlib.util.spec_from_file_location("banana2_batch_node", mod_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["banana2_batch_node"] = module
            spec.loader.exec_module(module)
            GeminiBatchNode = module.GeminiBatchNode

        # 构建传入 Banana2 的虚拟 kwargs（包含 prompt_1、image_1.x 以及 OSS 配置）
        virtual_kwargs = {
            "prompt_1": task["prompt"],
            "provider": config["provider"],
            "base_url": config["base_url"],
            "api_key": config.get("api_key", ""),
            "model": config.get("model", ""),
            "mode": config.get("mode", "Img2Img"),
            "aspect_ratio": config.get("aspect_ratio", "auto"),
            "response_format": config.get("response_format", "url"),
            "img_size": config.get("img_size", "2K"),
            "img_n": config.get("img_n", 1),
            "timeout": config.get("timeout", 200),
            "retry_count": config.get("retry_count", 0),
            "node_enabled": True,
            # OSS配置项（直接映射到 banana2 的 _parse_config 所期望的键名）
            "oss_endpoint": kwargs.get("oss_endpoint", ""),
            "oss_access_key_id": kwargs.get("oss_access_key_id", ""),
            "oss_access_key_secret": kwargs.get("oss_access_key_secret", ""),
            "oss_bucket_name": kwargs.get("oss_bucket_name", ""),
            "oss_object_prefix": kwargs.get("oss_object_prefix", "banana-images/"),
            "oss_use_signed_url": kwargs.get("oss_use_signed_url", True),
            "oss_signed_url_expire_seconds": kwargs.get("oss_signed_url_expire_seconds", 3600),
            "oss_security_token": kwargs.get("oss_security_token", "")
        }

        # 在Img2Img模式下才传递图片（最多10张）
        if config.get("mode", "Img2Img") == "Img2Img" and task.get("images"):
            for idx, img in enumerate(task["images"][:10], start=1):
                virtual_kwargs[f"image_1.{idx}"] = self._pil_to_tensor(img)

        # 调用 GeminiBatchNode 的解析逻辑，确保 OSS 配置被正确识别
        banana_node = GeminiBatchNode()
        parsed_config = banana_node._parse_config(virtual_kwargs)
        # 打印调试信息，展示解析后的 OSS 配置（避免使用 f-string 以防解析器问题）
        try:
            debug_val = parsed_config.get("oss_config")
            _dlog("[DEBUG] NanoBanana parsed_config oss_config:", debug_val)
        except Exception:
            pass

        # 解析任务并执行第一个任务
        tasks_parsed = banana_node._parse_tasks(virtual_kwargs, parsed_config)
        if tasks_parsed and tasks_parsed[0].get("is_valid"):
            result = await banana_node._execute_single_task_with_retry(tasks_parsed[0], parsed_config)
            return {
                "model": provider.capitalize(),
                "success": result.get("success", False),
                "image": result.get("image"),
                "url": result.get("url", ""),
                "status": result.get("response_code", 0),
                "info": result.get("info", "")
            }

        return {
            "model": provider.capitalize(),
            "success": False,
            "image": None,
            "url": "",
            "status": 0
        }

    async def _execute_doubao_model(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行豆包模型"""
        _dlog(f"[DOUBAO] 执行Doubao模型: {config.get('model', 'N/A')}")

        # 检查模型是否启用
        if not config.get("api_enabled", True):
            _dlog("[DOUBAO] Doubao模型已被禁用")
            return {
                "model": "Doubao",
                "success": False,
                "image": None,
                "url": "",
                "status": 0
            }

        # 动态导入 doubao_batch_node，兼容不同运行上下文（直接模块或作为包导入）
        try:
            import importlib
            DoubaoBatchNode = importlib.import_module("doubao_batch_node").DoubaoBatchNode
        except Exception:
            import importlib.util
            import sys
            import os
            mod_path = os.path.join(os.path.dirname(__file__), "doubao_batch_node.py")
            spec = importlib.util.spec_from_file_location("doubao_batch_node", mod_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["doubao_batch_node"] = module
            spec.loader.exec_module(module)
            DoubaoBatchNode = module.DoubaoBatchNode

        # 创建虚拟的单组任务
        virtual_kwargs = {
            "prompt_1": task["prompt"],
            "provider": config["provider"],
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "model": config["model"],
            "mode": config["mode"],
            "aspect_ratio": config["aspect_ratio"],
            "response_format": config["response_format"],
            "img_size": config["img_size"],
            "n": config["n"],
            "watermark": config["watermark"],
            "stream": config["stream"],
            "timeout": config["timeout"],
            "retry_count": config["retry_count"],
            "node_enabled": True
        }

        # 调试：打印 aspect_ratio 流向与图片信息，帮助排查比例计算问题
        try:
            _dlog(f"[DEBUG] Doubao virtual aspect_ratio (config): {config.get('aspect_ratio')}")
            _dlog(f"[DEBUG] Doubao virtual aspect_ratio (virtual_kwargs): {virtual_kwargs.get('aspect_ratio')}")
            if task.get("images"):
                for i, img in enumerate(task["images"], 1):
                    try:
                        _dlog(f"[DEBUG] Doubao task image{i} size: {img.size} mode: {img.mode}")
                    except Exception:
                        _dlog(f"[DEBUG] Doubao task image{i} type: {type(img)}")
            else:
                _dlog("[DEBUG] Doubao task has no images")
        except Exception:
            pass

        # 保持传入的 aspect_ratio（例如 "auto" 或指定比例），由子节点自行处理

        # 在Img2Img模式下才传递图片（最多传 4 张参考图到 image_1.1..image_1.4）
        if config["mode"] == "Img2Img" and task["images"]:
            for idx, img in enumerate(task["images"][:4], start=1):
                virtual_kwargs[f"image_1.{idx}"] = self._pil_to_tensor(img)

        # 使用DoubaoBatchNode的逻辑执行
        doubao_node = DoubaoBatchNode()
        tasks = doubao_node._parse_tasks(virtual_kwargs, config)
        if tasks and tasks[0]["is_valid"]:
            result = await doubao_node._execute_single_task_with_retry(tasks[0], config)
            return {
                "model": "Doubao",
                "success": result["success"],
                    "image": result.get("image"),
                    "url": result.get("url", ""),
                    "status": result.get("response_code", 0),
                    "info": result.get("info", "")
            }

        return {
            "model": "Doubao",
            "success": False,
            "image": None,
            "url": "",
            "status": 0
        }

    def _process_comparison_results(self, results: List[Dict[str, Any]]) -> Tuple:
        """处理比较结果"""
        # 解析结果
        banana_result = results[0] if len(results) > 0 else {"success": False, "image": None, "url": "", "status": 0}
        doubao_result = results[1] if len(results) > 1 else {"success": False, "image": None, "url": "", "status": 0}

        # 构建输出
        # ComfyUI图像格式: [B, H, W, C]
        empty_image = torch.zeros((1, 64, 64, 3))

        # banana结果
        if banana_result["success"]:
            if banana_result["image"]:
                # 有图片数据（Base64格式）
                banana_array = self._pil_to_tensor(banana_result["image"])
                if banana_array is not None:
                    banana_image = banana_array.unsqueeze(0)  # 添加批次维度 [1,H,W,C]
                    banana_image = banana_image  # 保持变量名
                    banana_status = banana_result["status"]
                else:
                    banana_image = empty_image
                    banana_status = 3
            else:
                # 无图片数据但有URL（URL格式）
                _dlog(f"[INFO] ModelCompare banana URL格式响应: {banana_result['url']}")
                banana_image = torch.full((1, 64, 64, 3), 0.5)  # 批次格式，灰色占位符，0-1范围
                banana_status = banana_result["status"]
        else:
            banana_image = empty_image
            banana_status = banana_result["status"]

        banana_url = banana_result["url"]

        # 豆包结果
        if doubao_result["success"]:
            if doubao_result["image"]:
                # 有图片数据（Base64格式）
                doubao_array = self._pil_to_tensor(doubao_result["image"])
                if doubao_array is not None:
                    doubao_image = doubao_array.unsqueeze(0)  # 添加批次维度 [1,H,W,C]
                    doubao_status = doubao_result["status"]
                else:
                    doubao_image = empty_image
                    doubao_status = 3
            else:
                # 无图片数据但有URL（URL格式）
                _dlog(f"[INFO] ModelCompare Doubao URL格式响应: {doubao_result['url']}")
                doubao_image = torch.full((1, 64, 64, 3), 0.5)  # 批次格式，灰色占位符，0-1范围
                doubao_status = doubao_result["status"]
        else:
            doubao_image = empty_image
            doubao_status = doubao_result["status"]

        doubao_url = doubao_result["url"]

        # 比较统计
        banana_success = 1 if banana_result["success"] else 0
        doubao_success = 1 if doubao_result["success"] else 0
        comparison_stats = f"banana:{banana_success}, Doubao:{doubao_success}"

        # per-model info fields (可能不存在)
        banana_info = banana_result.get("info", "")
        doubao_info = doubao_result.get("info", "")

        # 合并输出：仅包含成功的图片（以批次格式tensor表示）
        merged_images = []
        if banana_result.get("success") and banana_image is not None:
            merged_images.append(banana_image)
        if doubao_result.get("success") and doubao_image is not None:
            merged_images.append(doubao_image)
        if not merged_images:
            merged_images = [empty_image]

        # urls 和 responses 列表 -> JSON 字符串
        urls_json = json.dumps([banana_url or "", doubao_url or ""], ensure_ascii=False)
        responses_json = json.dumps([banana_status, doubao_status], ensure_ascii=False)

        # 返回：merged_images(list)，urls(json)，responses(json)，model1..., model2..., comparison_stats
        return (
            merged_images, urls_json, responses_json,
            banana_image, banana_url, banana_status, banana_info,
            doubao_image, doubao_url, doubao_status, doubao_info,
            comparison_stats
        )

    def _pil_to_tensor(self, image: Image.Image) -> Optional[torch.Tensor]:
        """将PIL图像转换为ComfyUI期望的torch.Tensor格式，带错误检查"""
        try:
            if image is None:
                print("[ERROR] ModelCompare输入图像为空")
                return None

            # 注意：图片已经在_download_image中验证过了，这里不再重复验证
            # 如果图片能到达这里，说明它已经是有效的PIL图像

            # 确保RGB模式
            if image.mode != "RGB":
                _dlog(f"[INFO] ModelCompare转换图片模式: {image.mode} -> RGB")
                image = image.convert("RGB")

            # 检查图片尺寸
            width, height = image.size
            if width == 0 or height == 0:
                print(f"[ERROR] ModelCompare图片尺寸无效: {width}x{height}")
                return None

            # 转换为numpy数组，保持0-255范围
            _dlog(f"[INFO] ModelCompare转换图片尺寸: {width}x{height}")
            np_img = np.array(image)

            # 检查数组形状
            if len(np_img.shape) != 3 or np_img.shape[2] != 3:
                print(f"[ERROR] ModelCompare图片数组形状异常: {np_img.shape}")
                return None

            # 转换为torch.Tensor，归一化到0-1范围，格式: [H, W, C] (ComfyUI标准格式)
            tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)

            _dlog(f"[SUCCESS] ModelCompare图片转换为torch.Tensor成功，形状: {tensor.shape}")
            return tensor

        except Exception as e:
            print(f"[ERROR] ModelCompare图片转tensor失败: {str(e)}")
            return None

    def _get_empty_outputs(self) -> Tuple:
        """返回空的输出"""
        # ComfyUI图像格式: torch.Tensor [B, H, W, C]，范围0-1
        empty_image = torch.zeros((1, 64, 64, 3))
        # 返回：merged_images(list)，urls(json)，responses(json)，model1_image,model1_url,model1_response,model1_info,model2_image,model2_url,model2_response,model2_info,comparison_stats
        return (
            [empty_image], "[]", "[]",  # merged
            empty_image, "", 0, "未执行",  # model1
            empty_image, "", 0, "未执行",  # model2
            "banana:0, Doubao:0"  # comparison_stats
        )


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "laowang_ModelCompare": ModelCompareNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "laowang_ModelCompare": "laowang_ModelCompare"
}
