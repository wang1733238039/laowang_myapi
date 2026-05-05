"""
DoubaoBatchNode ComfyUI Node
豆包批量并发绘图节点，支持多组任务并发处理
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


def _calculate_aspect_ratio(width: int, height: int) -> str:
    """计算图片的宽高比，返回最接近的NanoBanana支持的比例"""
    # 验证输入参数
    if width is None or height is None or width <= 0 or height <= 0:
        print(f"[警告] 无效的图片尺寸 width={width}, height={height}，使用默认比例 1:1")
        return "1:1"

    ratio = width / height
    print(f"[调试] 计算宽高比: {width}/{height} = {ratio:.6f}")

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
        print(f"[调试] 比例 {ratio_name} ({target_ratio:.6f}): 差值 = {diff:.6f}")
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio_name

    print(f"[调试] 最终匹配比例: {best_ratio} (差值 = {min_diff:.6f})")
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


class DoubaoBatchNode:
    """
    DoubaoBatchNode ComfyUI节点 - 支持并发多组任务处理豆包模型
    """
    CATEGORY = "laowang_myapi"

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入插槽"""
        required = {
            "provider": ("STRING", {
                "default": "comfly",
                "tooltip": "供应商名称"
            }),
            "base_url": ("STRING", {
                "default": "https://ai.comfly.chat",
                "tooltip": "API基础地址"
            }),
            "api_key": ("STRING", {
                "tooltip": "API密钥"
            }),
            "model": ("STRING", {
                "default": "doubao-seedream-4-5-251128",
                "tooltip": "豆包模型"
            }),
            "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                "default": "auto",
                "tooltip": "图像宽高比，auto模式自动根据输入图片比例调整"
            }),
            "mode": (["Text2Img", "Img2Img"], {
                "default": "Img2Img",
                "tooltip": "图像生成模式"
            }),
            "response_format": (["url", "b64_json"], {
                "default": "url",
                "tooltip": "响应格式"
            }),
            "img_size": (["2K","4K"], {
                "default": "2K",
                "tooltip": "图片尺寸 (豆包4.5仅支持2K和4K)"
            }),
            "n": ("INT", {
                "default": 1,
                "min": 1,
                "max": 4,
                "tooltip": "生成图片数量 (1-4)"
            }),
            "watermark": ("BOOLEAN", {
                "default": False,
                "tooltip": "是否添加水印"
            }),
            "stream": ("BOOLEAN", {
                "default": False,
                "tooltip": "是否流式响应"
            }),
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
        }

        # 可选的组输入（可以为None）
        optional = {}
        for group in range(1, 11):
            for img_idx in range(1, 11):
                optional[f"image_{group}.{img_idx}"] = ("IMAGE", {
                    "tooltip": f"组{group}的第{img_idx}张参考图像"
                })

            # prompt 强制为插槽（不在前端显示文本框），通过 forceInput=True 隐藏输入框，只保留插槽
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
 
    # 标记第一个输出为列表（images 返回 List[IMAGE]），其它保持单值
    OUTPUT_IS_LIST = (
        True,
        False, False,  # urls, responses
        # group1
                      False, False, False, False,
        # group2
                      False, False, False, False,
        # group3
                      False, False, False, False,
        # group4
                      False, False, False, False,
        # group5
                      False, False, False, False,
        # group6
                      False, False, False, False,
        # group7
        False, False, False, False,
        # group8
        False, False, False, False,
        # group9
        False, False, False, False,
        # group10
        False, False, False, False,
        # stats
        False,
    )

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
            print("\n[DEBUG] DoubaoBatchNode 执行开始 =====")
            print(f"[INFO] 节点启用状态: {kwargs.get('node_enabled', True)}")
            print(f"[INFO] 基础URL: {kwargs.get('base_url', 'N/A')}")
            print(f"[INFO] API密钥: {'已配置' if kwargs.get('api_key') else '未配置'}")
            print(f"[INFO] 模型: {kwargs.get('model', 'N/A')}")
            print(f"[INFO] 图片尺寸: {kwargs.get('img_size', 'N/A')}")
            print(f"[INFO] 图片数量: {kwargs.get('img_n', 'N/A')}")
            print(f"[INFO] 宽高比: {kwargs.get('aspect_ratio', 'N/A')}")
            print(f"[INFO] 响应格式: {kwargs.get('response_format', 'N/A')}")
            print(f"[INFO] 水印: {kwargs.get('watermark', 'N/A')}")
            print(f"[INFO] 流式输出: {kwargs.get('stream', 'N/A')}")
            print(f"[INFO] 并发数: {kwargs.get('concurrency', 'N/A')}")
            print(f"[INFO] 超时时间: {kwargs.get('timeout', 'N/A')}")
            print(f"[INFO] 重试次数: {kwargs.get('max_retries', 'N/A')}")

            # 显示各组的输入状态
            print("\n[DEBUG] 各组输入状态:")
            for group in range(1, 11):
                has_images = any(kwargs.get(f"image_{group}.{i}") is not None for i in range(1, 11))
                prompt = kwargs.get(f"prompt_{group}")
                print(f"  组{group}: 图片={has_images}, 提示词={'有' if prompt else '无'}")

            # 解析输入参数
            config = self._parse_config(kwargs)
            tasks = self._parse_tasks(kwargs, config)

            print(f"\n📊 解析结果: 共{len(tasks)}个任务, 其中{len([t for t in tasks if t['is_valid']])}个有效")
            print("=" * 50)

            # 过滤有效任务
            valid_tasks = [task for task in tasks if task["is_valid"]]

            if not valid_tasks:
                print("DoubaoBatch: 没有有效的任务组")
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
            print("\n[DEBUG] DoubaoBatchNode 执行结果汇总:")
            print(f"  [INFO] 总任务数: {len(valid_tasks)}")
            print(f"  [SUCCESS] 成功任务: {len([r for r in results if r.get('success', False)])}")
            print(f"  [ERROR] 失败任务: {len([r for r in results if not r.get('success', False)])}")

            for i, result in enumerate(results, 1):
                status = "[SUCCESS]" if result.get("success", False) else "[ERROR]"
                print(f"  任务{i}: {status} {result.get('info', '无信息')}")

            print("\n[DEBUG] 准备返回最终输出...")
            print("=" * 50)

            # 处理结果
            return self._process_results(results)

        except Exception as e:
            print(f"DoubaoBatch: 执行出错 - {str(e)}")
            return self._get_empty_outputs()

    def _parse_config(self, kwargs) -> Dict[str, Any]:
        """解析配置参数"""
        return {
            "provider": kwargs.get("provider", "comfly"),
            "base_url": kwargs.get("base_url", "https://ai.comfly.chat"),
            "api_key": kwargs.get("api_key", ""),
            "model": kwargs.get("model", "doubao-seedream-4-5-251128"),
            "mode": kwargs.get("mode", "Img2Img"),
            "aspect_ratio": kwargs.get("aspect_ratio", "auto"),
            "response_format": kwargs.get("response_format", "url"),
            "img_size": kwargs.get("img_size", "2K"),
            "n": kwargs.get("n", 1),
            "seed": kwargs.get("seed", 0),
            "watermark": kwargs.get("watermark", False),
            "stream": kwargs.get("stream", False),
            "timeout": kwargs.get("timeout", 200),
            "retry_count": kwargs.get("retry_count", 0)
        }

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
        # 文生图模式：该组prompt插槽(prompt_x)为空时候，该组任务不执行API任务
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
        print(f"DoubaoBatch: 开始执行 {len(tasks)} 个任务，使用 {max_concurrent} 个并发")

        async def execute_single_task(task):
            async with semaphore:
                return await self._execute_single_task_with_retry(task, config)

        # 并发执行所有任务
        results = await asyncio.gather(*[execute_single_task(task) for task in tasks], return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"DoubaoBatch: 任务{tasks[i]['group_id']}执行异常 - {str(result)}")
                processed_results.append({
                    "group_id": tasks[i]["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,  # 失败
                    "info": f"执行异常: {str(result)}"
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task_with_retry(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务（带重试）"""
        retry_count = config["retry_count"]

        for attempt in range(retry_count + 1):
            try:
                result = await self._execute_single_task(task, config)
                if result["success"]:
                    return result
                elif attempt < retry_count:
                    await asyncio.sleep(2)  # 重试间隔
                    continue
                else:
                    return result
            except Exception as e:
                if attempt < retry_count:
                    print(f"DoubaoBatch: 任务{task['group_id']}第{attempt+1}次尝试失败 - {str(e)}，准备重试")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"DoubaoBatch: 任务{task['group_id']}最终失败 - {str(e)}")
                    return {
                        "group_id": task["group_id"],
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": f"重试{retry_count}次后仍然失败，最后一次错误: {str(e)}"
                    }

    async def _execute_single_task(self, task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        # 构建API请求
        api_url, headers, payload = self._build_api_request(task, config)

        # ===== 调试信息: API请求详情 =====
        print(f"\n[DEBUG] 任务{task['group_id']} API请求构建:")
        print(f"  [URL] 请求URL: {api_url}")
        print(f"  [HEADERS] 请求头: {headers}")
        print(f"  [PAYLOAD] 请求体: {self._mask_b64_json(payload)}")
        print(f"  [IMAGES] 参考图片数量: {len(task['images'])}")
        print(f"  [PROMPT] 提示词: {task['prompt'][:100]}{'...' if len(task['prompt']) > 100 else ''}")
        print("-" * 30)

        # 发送请求
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(api_url, headers=headers, json=payload, timeout=config["timeout"])
            )

            if response.status_code == 200:
                result_data = response.json()

                # ===== 调试信息: API响应详情 =====
                print(f"[SUCCESS] 任务{task['group_id']} API响应成功:")
                print(f"  [STATUS] 响应状态码: {response.status_code}")
                print(f"  [RESPONSE] 响应数据: {result_data}")
                response_mode = "异步" if "task_id" in result_data else "同步"
                print(f"  [MODE] 响应模式: {response_mode}")
                print("-" * 30)

                # 检查是否是异步响应（包含task_id）还是同步响应（直接包含data）
                if "task_id" in result_data:
                    # 异步模式：获取task_id后轮询状态
                    return await self._handle_doubao_async_response(task["group_id"], result_data, config)
                elif "data" in result_data and isinstance(result_data["data"], list):
                    # 同步模式：直接处理结果
                    return self._parse_doubao_sync_response(task["group_id"], result_data, config["response_format"])
                else:
                    print(f"DoubaoBatch: 任务{task['group_id']} 未知响应格式: {result_data}")
                    return {
                        "group_id": task["group_id"],
                        "success": False,
                        "image": None,
                        "url": "",
                        "response_code": 2,
                        "info": f"未知响应格式: {str(result_data)}"
                    }
            else:
                print(f"DoubaoBatch: 任务{task['group_id']} API请求失败 - {response.status_code}: {response.text}")
                return {
                    "group_id": task["group_id"],
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": f"API请求失败 - {response.status_code}: {response.text}"
                }

        except requests.exceptions.Timeout:
            print(f"DoubaoBatch: 任务{task['group_id']} 请求超时")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"请求超时 ({config['timeout']}秒)"
            }
        except Exception as e:
            print(f"DoubaoBatch: 任务{task['group_id']} 请求异常 - {str(e)}")
            return {
                "group_id": task["group_id"],
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"请求异常: {str(e)}"
            }

    async def _upload_image_to_doubao(self, image: Image.Image, config: Dict[str, Any]) -> Optional[str]:
        """上传图片到豆包服务器获取URL"""
        try:
            # 将PIL图像转换为bytes
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            # 豆包图片上传API - 使用原生接口端点
            upload_url = "https://ark.cn-beijing.volces.com/api/v3/images/uploads"

            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "multipart/form-data"
            }

            # 构建multipart数据
            files = {
                "file": ("image.png", image_bytes, "image/png")
            }

            # 同步上传（豆包可能不支持异步上传）
            response = self.session.post(upload_url, headers=headers, files=files, timeout=30)

            if response.status_code == 200:
                upload_data = response.json()
                if "data" in upload_data and "url" in upload_data["data"]:
                    image_url = upload_data["data"]["url"]
                    print(f"DoubaoBatch: 图片上传成功 - {image_url}")
                    return image_url

            print(f"DoubaoBatch: 图片上传失败 - {response.status_code}: {response.text}")
            return None

        except Exception as e:
            print(f"DoubaoBatch: 图片上传异常 - {str(e)}")
            return None

    def _build_api_request(self, task: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """构建豆包API请求"""
        base_url = config["base_url"].rstrip("/")
        has_images = len(task["images"]) > 0

        # 处理aspect_ratio的auto模式
        final_aspect_ratio = config["aspect_ratio"]
        enhanced_prompt = task["prompt"]

        if config["aspect_ratio"] == "auto":
            if has_images:
                # 获取第一张图片的尺寸
                first_image = task["images"][0]
                width, height = _get_image_size_with_exif(first_image)
                if width and height:
                    final_aspect_ratio = _calculate_aspect_ratio(width, height)
                    print(f"[AUTO] 根据输入图片({width}x{height})计算比例: {final_aspect_ratio}")
                else:
                    final_aspect_ratio = "1:1"
                    print("[AUTO] 无法获取图片尺寸，使用默认比例: 1:1")
            else:
                final_aspect_ratio = "1:1"
                print("[AUTO] 无输入图片，使用默认比例: 1:1")

        # 根据aspect_ratio调整prompt，添加比例描述
        # 注意：即使是1:1比例也需要明确指定，否则豆包可能使用默认比例(3:4)
        if True:  # 总是添加比例描述，确保豆包按指定比例生成
            # 将比例转换为自然语言描述
            ratio_descriptions = {
                "1:1": "1:1正方形比例",
                "2:3": "2:3竖屏比例",
                "3:2": "3:2横屏比例",
                "3:4": "竖屏3:4比例",
                "4:3": "横屏4:3比例",
                "4:5": "竖屏4:5比例",
                "5:4": "横屏5:4比例",
                "9:16": "竖屏9:16比例",
                "16:9": "横屏16:9宽屏比例",
                "21:9": "超宽屏21:9比例"
            }
            ratio_desc = ratio_descriptions.get(final_aspect_ratio, f"{final_aspect_ratio}比例")
            enhanced_prompt = f"{task['prompt']}，图片比例为{ratio_desc}"
            print(f"[ASPECT_RATIO] 增强prompt添加比例描述: {ratio_desc}")

        # 根据mode决定是否使用图像
        use_images = has_images and config["mode"] == "Img2Img"

        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        # 豆包API基础URL - 支持异步模式
        api_url = f"{base_url}/v1/images/generations"
        if config.get("async_mode", True):  # 默认使用异步模式
            api_url += "?async=true"

        # 根据img_size和计算出的比例确定最终的size参数
        final_size = config["img_size"]
        if config["img_size"] in ["2K", "4K"] and final_aspect_ratio != "auto":
            # 对于2K/4K字符串尺寸，根据比例使用推荐的像素值以获得更精确的控制
            size_recommendations = {
                "1:1": {"2K": "2048x2048", "4K": "4096x4096"},
                "4:3": {"2K": "2304x1728", "4K": "3072x2304"},
                "3:4": {"2K": "1728x2304", "4K": "2304x3072"},
                "16:9": {"2K": "2560x1440", "4K": "3840x2160"},
                "9:16": {"2K": "1440x2560", "4K": "2160x3840"},
                "3:2": {"2K": "2496x1664", "4K": "3328x2240"},
                "2:3": {"2K": "1664x2496", "4K": "2240x3328"},
                "21:9": {"2K": "3024x1296", "4K": "4032x1728"},
                "5:4": {"2K": "2560x2048", "4K": "3840x3072"},
                "4:5": {"2K": "2048x2560", "4K": "3072x3840"},
            }
            if final_aspect_ratio in size_recommendations and config["img_size"] in size_recommendations[final_aspect_ratio]:
                final_size = size_recommendations[final_aspect_ratio][config["img_size"]]
                print(f"[SIZE] 使用推荐像素值: {final_size} (基于{config['img_size']}和比例{final_aspect_ratio})")
            else:
                print(f"[SIZE] 使用字符串尺寸: {final_size} (比例{final_aspect_ratio}无推荐值)")

        # 基础payload - 遵循Seedream-4.5原生接口
        payload = {
            "model": config["model"],
            "prompt": enhanced_prompt,  # 使用增强后的prompt（包含比例描述）
            "response_format": config.get("response_format", "url"),
            "size": final_size,  # 使用最终确定的尺寸规格
            "watermark": config["watermark"],
            "stream": config["stream"],
            "sequential_image_generation": "disabled"  # 默认关闭组图功能，生成单图
        }

        # 如果n > 1，启用组图功能；同时始终包含 n 参数（部分供应商/示例同时需要 n）
        payload["n"] = config["n"]
        if config["n"] > 1:
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {
                "max_images": config["n"]  # 设置最大图片数量
            }

        # 如果有图片，添加到image数组
        if use_images:
            # 处理多图输入 - 上传图片到豆包服务器或转换为可访问的URL
            image_urls = []

            # 限制图片数量：Seedream-4.5最多支持14张参考图
            max_images = min(len(task["images"]), 14)

            for i in range(max_images):
                img = task["images"][i]
                # 直接转换为Base64格式（豆包API支持base64格式的图片）
                try:
                    import base64
                    from io import BytesIO

                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_urls.append(f"data:image/png;base64,{img_base64}")
                    print(f"Doubao: 图片{i+1}转换为Base64格式")
                except Exception as e:
                    print(f"Doubao: 图片{i+1}处理失败: {e}")
                    continue

            if image_urls:
                payload["image"] = image_urls
                print(f"Doubao: 使用 {len(image_urls)} 张参考图片")
            else:
                print("Doubao: 所有图片处理失败，将作为文生图处理")

        return api_url, headers, payload

    async def _handle_doubao_async_response(self, group_id: int, response_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """处理豆包异步响应，获取task_id并轮询状态"""
        try:
            # 从响应中获取task_id
            task_id = None
            if "task_id" in response_data:
                # 直接在响应根层级
                task_id = response_data["task_id"]
            elif "data" in response_data and isinstance(response_data["data"], str):
                # 在data字段中
                task_id = response_data["data"]

            if task_id:
                print(f"DoubaoBatch: 任务{group_id} 异步任务已提交，task_id: {task_id}")
                # 开始轮询查询状态
                return await self._poll_doubao_task_status(group_id, task_id, config)
            else:
                print(f"DoubaoBatch: 任务{group_id} 异步响应中未找到task_id: {response_data}")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": f"异步响应中未找到task_id: {str(response_data)}"
                }

        except Exception as e:
            print(f"DoubaoBatch: 任务{group_id} 处理异步响应异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"处理异步响应异常: {str(e)}"
            }

    def _parse_doubao_sync_response(self, group_id: int, response_data: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析豆包同步响应 - 支持单图和组图"""
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                # 检查是否有成功的图片
                successful_images = []
                successful_urls = []

                for image_data in response_data["data"]:
                    # 检查是否是错误信息
                    if "error" in image_data:
                        error_info = image_data["error"]
                        print(f"DoubaoBatch: 任务{group_id} 图片生成失败 - {error_info.get('message', 'Unknown error')}")
                        continue

                    # 提取URL
                    image_url = image_data.get("url", "")
                    if image_url:
                        # 下载图像
                        image = self._download_image(image_url)
                        if image:
                            successful_images.append(image)
                            successful_urls.append(image_url)

                # 如果有成功的图片，返回第一张
                if successful_images:
                    print(f"DoubaoBatch: 任务{group_id} 成功生成 {len(successful_images)} 张图片")
                    # 根据response_format决定URL返回值
                    return_url = "b64_ok" if response_format == "b64_json" else successful_urls[0]
                    return {
                        "group_id": group_id,
                        "success": True,
                        "image": successful_images[0],  # 返回第一张图片
                        "url": return_url,
                        "response_code": 1,
                        "info": f"成功生成 {len(successful_images)} 张图片"
                    }

            # 检查是否有顶层错误
            if "error" in response_data:
                error_info = response_data["error"]
                error_msg = error_info.get("message", "Unknown error")
                print(f"DoubaoBatch: 任务{group_id} API错误 - {error_msg}")
                return {
                    "group_id": group_id,
                    "success": False,
                    "image": None,
                    "url": "",
                    "response_code": 2,
                    "info": f"API错误: {error_msg}"
                }

            print(f"DoubaoBatch: 任务{group_id} 同步响应解析失败 - {response_data}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"同步响应解析失败: {str(response_data)}"
            }

        except Exception as e:
            print(f"DoubaoBatch: 任务{group_id} 同步响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"同步响应解析异常: {str(e)}"
            }

    async def _poll_doubao_task_status(self, group_id: int, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """轮询查询豆包任务状态，每5秒查询一次"""
        base_url = config["base_url"].rstrip("/").replace("?async=true", "")  # 移除async参数
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }

        max_polls = 60  # 最多轮询60次（5分钟）
        poll_count = 0

        while poll_count < max_polls:
            poll_count += 1

            try:
                # 构建查询URL
                query_url = f"{base_url}/v1/images/tasks/{task_id}"

                # 发送查询请求
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(query_url, headers=headers, timeout=30)
                )

                if response.status_code == 200:
                    status_data = response.json()

                    if "data" in status_data:
                        task_info = status_data["data"]
                        status = task_info.get("status", "")
                        progress = task_info.get("progress", "0%")

                        print(f"DoubaoBatch: 任务{group_id} 状态查询 [{poll_count}] - 状态: {status}, 进度: {progress}")

                        if status == "SUCCESS":
                            # 任务完成，解析结果
                            return self._parse_doubao_success_response(group_id, task_info, config["response_format"])

                        elif status == "FAILURE":
                            # 任务失败
                            fail_reason = task_info.get("fail_reason", "未知错误")
                            print(f"DoubaoBatch: 任务{group_id} 生成失败 - {fail_reason}")
                            return {
                                "group_id": group_id,
                                "success": False,
                                "image": None,
                                "url": "",
                                "response_code": 2,
                                "info": f"API返回失败: {fail_reason}"
                            }

                        elif status in ["IN_PROGRESS", "NOT_START", "PENDING"]:
                            # 任务进行中，继续等待
                            await asyncio.sleep(5)  # 等待5秒
                            continue

                        else:
                            print(f"DoubaoBatch: 任务{group_id} 未知状态: {status}")
                            await asyncio.sleep(5)
                            continue

                    else:
                        print(f"DoubaoBatch: 任务{group_id} 状态查询响应格式错误: {status_data}")
                        await asyncio.sleep(5)
                        continue

                else:
                    print(f"DoubaoBatch: 任务{group_id} 状态查询失败 - {response.status_code}: {response.text}")
                    await asyncio.sleep(5)
                    continue

            except Exception as e:
                print(f"DoubaoBatch: 任务{group_id} 状态查询异常 - {str(e)}")
                await asyncio.sleep(5)
                continue

        # 超时
        print(f"DoubaoBatch: 任务{group_id} 查询超时，已等待{max_polls * 5}秒")
        return {
            "group_id": group_id,
            "success": False,
            "image": None,
            "url": "",
            "response_code": 2,
            "info": f"异步查询超时，已等待{max_polls * 5}秒"
        }

    def _parse_doubao_success_response(self, group_id: int, task_info: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """解析豆包异步成功的响应 - 支持单图和组图"""
        try:
            if "data" in task_info and "data" in task_info["data"]:
                # 检查是否有成功的图片
                successful_images = []
                successful_urls = []

                for image_data in task_info["data"]["data"]:
                    # 检查是否是错误信息
                    if "error" in image_data:
                        error_info = image_data["error"]
                        print(f"DoubaoBatch: 任务{group_id} 图片生成失败 - {error_info.get('message', 'Unknown error')}")
                        continue

                    # 提取URL或base64数据
                    image_url = image_data.get("url", "")
                    b64_data = image_data.get("b64_json", "")

                    # 优先使用URL，如果URL为空则使用b64_json
                    if image_url:
                        final_url = image_url
                    elif b64_data:
                        # 将b64_json转换为data URL格式
                        final_url = f"data:image/png;base64,{b64_data}"
                    else:
                        print(f"DoubaoBatch: 任务{group_id} 图片数据为空，跳过")
                        continue

                    if final_url:
                        # 根据URL格式决定是否下载图片
                        if final_url.startswith("data:image"):
                            # base64格式，需要下载转换
                            image = self._download_image(final_url)
                            if image:
                                successful_images.append(image)
                                successful_urls.append(final_url)
                        else:
                            # URL格式，直接添加到URL列表
                            successful_images.append(None)  # URL格式不下载图片
                            successful_urls.append(final_url)

                # 如果有成功的图片或URL，返回第一张
                if successful_images or successful_urls:
                    success_count = len([img for img in successful_images if img is not None]) + len([url for url in successful_urls if not url.startswith("data:image")])
                    print(f"DoubaoBatch: 任务{group_id} 成功生成 {success_count} 张图片/URL")

                    # 返回第一张成功的图片或URL
                    first_image = None
                    first_url = ""
                    for i, img in enumerate(successful_images):
                        if img is not None:
                            first_image = img
                            first_url = successful_urls[i]
                            break
                        elif successful_urls[i] and not successful_urls[i].startswith("data:image"):
                            first_url = successful_urls[i]
                            break

                    # 根据response_format决定URL返回值
                    return_url = "b64_ok" if response_format == "b64_json" else first_url
                    return {
                        "group_id": group_id,
                        "success": True,
                        "image": first_image,  # 可能为None（URL格式）
                        "url": return_url,
                        "response_code": 1,
                        "info": f"成功生成 {success_count} 张图片/URL | API响应: {self._mask_b64_json(task_info)}"
                    }

            print(f"DoubaoBatch: 任务{group_id} 异步响应解析失败 - {task_info}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"异步响应解析失败: {str(task_info)}"
            }

        except Exception as e:
            print(f"DoubaoBatch: 任务{group_id} 异步响应解析异常 - {str(e)}")
            return {
                "group_id": group_id,
                "success": False,
                "image": None,
                "url": "",
                "response_code": 2,
                "info": f"异步响应解析异常: {str(e)}"
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

                    # 对于base64数据，如果能成功打开图片，说明数据完整
                    # 不需要额外的verify()验证（verify()会关闭图片对象）
                    print(f"[SUCCESS] Doubao Base64图片处理成功")
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

                        print(f"[SUCCESS] Doubao URL图片下载并验证成功，大小: {len(response.content)} bytes")
                        return img
                    else:
                        print(f"[ERROR] Doubao图片下载失败，状态码: {response.status_code}")

            except Exception as e:
                print(f"[ERROR] Doubao图片处理失败 (尝试 {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    import time
                    time.sleep(1)  # 等待1秒后重试
                    continue

        print(f"[ERROR] Doubao图片下载失败，已重试 {max_retries + 1} 次")
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
                            # 成功图片列表存储批次格式的tensor
                            successful_images.append(group_image)
                            group_outputs.extend([
                                group_image,
                                result["url"],
                                result["response_code"],
                                result.get("info", "成功")
                            ])
                        else:
                            # 图片转换失败，当作失败处理
                            print(f"[ERROR] Doubao任务{group_id} 图片转换失败，使用空图片")
                            empty_image = torch.zeros((1, 64, 64, 3))
                            group_outputs.extend([
                                empty_image,
                                result["url"],
                                3,  # 转换失败
                                "图片转换失败"
                            ])
                    else:
                        # 无图片数据但有URL（URL格式），创建占位符图像
                        print(f"[INFO] Doubao任务{group_id} URL格式响应: {result['url']}")
                        # 为URL格式创建一个特殊的占位符图像（批次格式），表示这是URL链接
                        url_placeholder = torch.full((1, 64, 64, 3), 0.5)  # 批次格式，灰色占位符，0-1范围
                        successful_images.append(url_placeholder)
                        # group_outputs 使用批次格式占位符
                        group_placeholder = url_placeholder
                        group_outputs.extend([
                            group_placeholder,
                            result["url"],
                            result["response_code"],
                            result.get("info", "URL格式响应")
                        ])
                else:
                    # 失败情况下的独立组输出
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
                empty_image = torch.zeros((1, 64, 64, 3))
                group_outputs.extend([
                    empty_image,
                    "",
                    0,
                    "未执行的任务"
                ])

        # 合并输出images：返回图像列表（List of tensors，元素为批次格式 [1,H,W,C]）
        if successful_images:
            # successful_images 中每个元素已经是批次格式 [1,H,W,C]
            merged_images = successful_images  # List of tensors
            print(f"[DEBUG] Doubao合并图像列表长度: {len(merged_images)}")
            for i, img in enumerate(merged_images):
                print(f"  图片{i+1} 形状: {img.shape}")
        else:
            # 如果没有成功的图像，返回包含单个占位符（批次格式）的列表
            empty_image = torch.full((1, 64, 64, 3), 0.5)  # 灰色占位符 [1, H, W, C]
            merged_images = [empty_image]
            print(f"[DEBUG] Doubao空合并图像列表 (占位符)")

        # urls和responses作为JSON字符串
        urls_json = json.dumps(all_urls, ensure_ascii=False)
        responses_json = json.dumps(all_responses, ensure_ascii=False)

        # 统计输出
        stats = f"(有效任务:{valid_tasks}, 成功任务:{success_tasks})"

        # 返回所有输出：合并输出(3) + 独立组输出(30) + 统计输出(1) = 34个
        return tuple([merged_images, urls_json, responses_json] + group_outputs + [stats])

    def _mask_b64_json(self, data: Any) -> Any:
        """屏蔽API响应中的b64_json和base64图片内容以避免日志溢出"""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key == "b64_json" and isinstance(value, str) and len(value) > 20:
                    # 只保留前20个字符，并显示数据长度
                    data_length = len(value)
                    masked[key] = f"{value[:20]}...[BASE64_DATA_{data_length}_CHARS]"
                elif key == "image" and isinstance(value, list):
                    # 处理image字段中的base64数据
                    masked_images = []
                    for img in value:
                        if isinstance(img, str) and img.startswith("data:image") and len(img) > 50:
                            # 截断base64图片数据
                            data_length = len(img)
                            masked_images.append(f"{img[:50]}...[BASE64_IMAGE_{data_length}_CHARS]")
                        else:
                            masked_images.append(img)
                    masked[key] = masked_images
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
                print("[ERROR] Doubao输入图像为空")
                return None

            # 注意：图片已经在_download_image中验证过了，这里不再重复验证
            # 如果图片能到达这里，说明它已经是有效的PIL图像

            # 确保RGB模式
            if image.mode != "RGB":
                print(f"[INFO] Doubao转换图片模式: {image.mode} -> RGB")
                image = image.convert("RGB")

            # 检查图片尺寸
            width, height = image.size
            if width == 0 or height == 0:
                print(f"[ERROR] Doubao图片尺寸无效: {width}x{height}")
                return None

            # 转换为numpy数组，保持0-255范围
            print(f"[INFO] Doubao转换图片尺寸: {width}x{height}")
            np_img = np.array(image)

            # 检查数组形状
            if len(np_img.shape) != 3 or np_img.shape[2] != 3:
                print(f"[ERROR] Doubao图片数组形状异常: {np_img.shape}")
                return None

            # 转换为torch.Tensor，归一化到0-1范围，格式: [H, W, C] (ComfyUI标准格式)
            tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)

            print(f"[SUCCESS] Doubao图片转换为torch.Tensor成功，形状: {tensor.shape}")
            return tensor

        except Exception as e:
            print(f"[ERROR] Doubao图片转tensor失败: {str(e)}")
            return None

    def _get_empty_outputs(self) -> Tuple:
        """返回空的输出"""
        # ComfyUI图像格式: torch.Tensor [B, H, W, C]，范围0-1
        empty_image = torch.zeros((1, 64, 64, 3))

        # 合并输出 - images 为列表格式，返回包含单个空图像的列表
        merged_outputs = [[empty_image], "[]", "[]"]

        # 独立组输出 (10组 × 4)
        group_outputs = []
        for _ in range(10):
            group_outputs.extend([empty_image, "", 0, "未执行的任务"])

        # 统计输出
        stats_output = ["(有效任务:0, 成功任务:0)"]

        return tuple(merged_outputs + group_outputs + stats_output)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "DoubaoBatch": DoubaoBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoBatch": "laowang_DoubaoBatch"
}
