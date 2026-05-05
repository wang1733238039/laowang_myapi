"""
AIYang ComfyUI API Nodes
自定义ComfyUI节点，支持多种AI图像生成API
"""

import traceback

from .banana2_batch_node import NODE_CLASS_MAPPINGS as BANANA2_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BANANA2_DISPLAY
from .doubao_batch_node import NODE_CLASS_MAPPINGS as DOUBAO_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DOUBAO_DISPLAY
from .model_compare_node import NODE_CLASS_MAPPINGS as COMPARE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as COMPARE_DISPLAY

# 初始化主字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 合并所有节点映射（以后注册的覆盖前面的同名键）
for key, value in  BANANA2_MAPPINGS.items():
    NODE_CLASS_MAPPINGS[key] = value
for key, value in  BANANA2_DISPLAY.items():
    NODE_DISPLAY_NAME_MAPPINGS[key] = value

for key, value in DOUBAO_MAPPINGS.items():
    NODE_CLASS_MAPPINGS[key] = value

for key, value in COMPARE_MAPPINGS.items():
    NODE_CLASS_MAPPINGS[key] = value

for key, value in DOUBAO_DISPLAY.items():
    NODE_DISPLAY_NAME_MAPPINGS[key] = value

for key, value in COMPARE_DISPLAY.items():
    NODE_DISPLAY_NAME_MAPPINGS[key] = value

# 旧 class key 向后兼容别名（守卫式注入）
# 老版本曾使用不带前缀的 key，加上前缀后通过别名保持已有 workflow 可用。
# `_old not in NODE_CLASS_MAPPINGS` 守卫确保不会反向覆盖其他插件的同名节点。
_LEGACY_ALIASES = {
    "GeminiBatch": "laowang_GeminiBatch",
    "DoubaoBatch": "laowang_DoubaoBatch",
    "ModelCompare": "laowang_ModelCompare",
}
for _old, _new in _LEGACY_ALIASES.items():
    if _new in NODE_CLASS_MAPPINGS and _old not in NODE_CLASS_MAPPINGS:
        NODE_CLASS_MAPPINGS[_old] = NODE_CLASS_MAPPINGS[_new]
        NODE_DISPLAY_NAME_MAPPINGS.setdefault(
            _old, NODE_DISPLAY_NAME_MAPPINGS.get(_new, _old)
        )

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("laowang_myapi ComfyUI Nodes loaded successfully!")
