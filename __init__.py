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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("AIYang ComfyUI API Nodes loaded successfully!")
print(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
print(f"Display name mappings: {NODE_DISPLAY_NAME_MAPPINGS}")
