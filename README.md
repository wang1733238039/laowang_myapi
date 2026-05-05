# AIYang ComfyUI API Nodes

基于ComfyUI的自定义节点，支持多种AI图像生成API的并发调用。采用模块化架构，按并发场景拆分为专门的节点，支持双供应商自动切换。

## 🚀 功能特性

- **🎯 核心要点**:
  - 双供应商自动适配：Comfly和BW无缝切换
  - 智能API路由：根据供应商+模型自动选择正确的接口
  - 异步并发处理：最大5并发，完美支持长时间生成任务
  - OSS图片上传：阿里云OSS集成，解决图片传输问题

- **🏗️ 模块化架构**: 按并发场景拆分为专门的节点，避免功能混乱
- **⚡ 三种并发模式**:
  - 多组图 + 单模型并发 (AIYang007_Banana2Batch)
  - 多组图 + 豆包模型并发 (AIYang007_DoubaoBatch)
  - 单组图 + 多模型比较 (AIYang007_ModelCompare)
- **🔄 三供应商支持**: Comfly + BW + grsai，自动适配API格式
- **🧠 智能任务调度**: 自动识别有效任务，支持文生图、图生图模式
- **🛡️ 完善的错误处理**: 支持超时重试，详细的状态反馈和错误隔离
- **🎨 ComfyUI原生支持**: 完全兼容ComfyUI的工作流系统和图像处理

## 📦 安装方法

1. 将整个项目文件夹复制到ComfyUI的`custom_nodes`目录下：
   ```
   ComfyUI/
   ├── custom_nodes/
   │   └── AIYang_comfyui_myapi/
   │       ├── __init__.py
   │       ├── banana2_batch_node.py
   │       ├── doubao_batch_node.py
   │       ├── model_compare_node.py
   │       ├── requirements.txt
   │       └── README.md
   ```

2. 安装依赖（如果需要）：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置API密钥：
   - 编辑`自留/config.py`文件，设置API密钥
   - 或在节点参数中直接输入API密钥

4. **完全重启ComfyUI**（重要！），节点将自动加载到"AIYang007_myapi"分类中

## 🚀 快速开始

1. **选择节点**: 在ComfyUI中找到"AIYang007_Banana2Batch"节点
2. **配置供应商**: 设置provider为"comfly"、"BW"或"grsai"
3. **输入API密钥**: 在api_key参数中填入对应的密钥
4. **选择模型**: 根据供应商选择合适的模型
5. **输入提示词**: 在prompt_1中输入生成描述
6. **执行生成**: 点击"Queue"开始异步并发生成

## 🎯 节点使用指南

### 节点总览

| 节点名称 | 功能场景 | 并发模式 | 支持供应商 | 支持模型 |
|---------|---------|---------|---------|---------|
| **AIYang007_Banana2Batch** | 多组图 + 单模型并发 | 10组任务并发 | comfly, BW, grsai | nano-banana系列 |
| **AIYang007_DoubaoBatch** | 多组图 + 单豆包模型并发 | 10组任务并发 | comfly | doubao-seedream系列 |
| **AIYang007_ModelCompare** | 单组图 + 多模型比较 | 双模型并发 | comfly, BW | banana + doubao |

### 节点位置
在ComfyUI中搜索对应节点名称，或在 "AIYang007_myapi" 分类下找到节点

### 输入参数

#### 图像输入 (40个插槽)
- `image_1.1` 到 `image_10.4`: 每组最多4张参考图像
- 支持PNG、JPEG等常见图像格式

#### 文本输入 (10个插槽)
- `prompt_1` 到 `prompt_10`: 每组的文本提示词

#### 配置参数
- `provider`: API供应商选择
  - "comfly": Comfly平台 (默认)
  - "BW": BananaWebAPI平台
- `base_url`: API基础地址
  - Comfly: "https://ai.comfly.chat"
  - BananaWebAPI: 对应平台的API地址
- `api_key`: API密钥 (必填，支持直接输入)
- `model`: 模型选择
  - Comfly: nano-banana系列、doubao-seedream系列
  - BananaWebAPI: nano-banana系列
- `mode`: 生成模式 ("Text2Img" 或 "Img2Img")
- `aspect_ratio`: 图像宽高比 ("auto", "1:1", "16:9"等)
- `response_format`: 响应格式 ("url" 或 "b64_json")
- `img_size`: 图片尺寸 ("1K", "2K", "4K")
- `img_n`: 生成图片数量 (1-1，仅Banana2Batch支持)
- `timeout`: 单次请求超时时间(秒) (默认: 200)
- `retry_count`: 失败重试次数 (默认: 0)
- `node_enabled`: 节点开关 (默认: True)

#### OSS配置 (可选，用于图片上传)
- `oss_endpoint`: OSS端点地址
- `oss_access_key_id`: OSS访问密钥ID
- `oss_access_key_secret`: OSS访问密钥Secret
- `oss_bucket_name`: OSS存储桶名称
- `oss_object_prefix`: OSS对象前缀
- `oss_use_signed_url`: 是否使用签名URL
- `oss_signed_url_expire_seconds`: 签名URL过期时间

### 输出结果

#### 合并输出 (3个)
- `images`: 所有成功生成的图像 (IMAGE类型)
- `urls`: 各组图像URL链接 (JSON字符串)
- `responses`: 各组执行状态 (JSON数组: 1=成功, 2=失败, 0=未执行)

#### 独立组输出 (30个)
- `group1_image` 到 `group10_image`: 每组生成的图像
- `group1_url` 到 `group10_url`: 每组图像URL
- `group1_response` 到 `group10_response`: 每组状态码

#### 统计输出 (1个)
- `stats`: 执行统计信息 `(有效任务:X, 成功任务:Y)`

### 使用示例

#### AIYang007_Banana2Batch (多组图 + 单模型并发)
**适用场景**: 批量生成多个不同主题的图像

1. 选择供应商: "comfly" 或 "BW"
2. 配置模型:
   - Comfly: nano-banana-2, nano-banana-pro等
   - BananaWebAPI: nano-banana系列模型
3. 设置参数: aspect_ratio, img_size, timeout等
4. 输入多组提示词: prompt_1 到 prompt_10
5. 可选添加参考图像: image_1.1 到 image_10.4 (图生图模式)
6. 执行节点，获取10组并发结果

**OSS上传**: 如果配置了OSS参数，图片会自动上传到OSS获取URL

#### AIYang007_DoubaoBatch (多组图 + 单豆包模型并发)
**适用场景**: 使用豆包模型批量生成高清图像

1. 选择供应商: "comfly" (豆包模型只支持Comfly)
2. 配置模型: doubao-seedream-4-5-251128 等
3. 设置豆包专用参数:
   - `n`: 生成图片数量 (1-4)
   - `img_size`: 图片尺寸 ("2K", "4K")
   - `watermark`: 是否添加水印
   - `stream`: 是否流式响应
4. 输入多组提示词: prompt_1 到 prompt_10
5. 可选添加参考图像URL (豆包支持URL引用)
6. 执行节点，获取豆包风格的批量高清图像

#### AIYang007_ModelCompare (单组图 + 多模型比较)
**适用场景**: 比较不同模型的生成效果

1. 输入单组参考图像和提示词 (image_1 到 image_4, prompt)
2. 配置两个模型:
   - 模型1: Comfly的nano-banana系列
   - 模型2: Comfly的doubao-seedream系列
3. 设置对比参数: 相同的aspect_ratio, img_size等
4. 执行节点，同时获得两个模型的生成结果进行对比
5. 输出包含两个模型的图像、URL和状态信息

#### 即梦4使用示例
1. 选择模型: `doubao-seedream-4-0-250828`
2. 设置参数:
   - `n`: 3 (生成3张图片)
   - `size`: "2K" (2K分辨率)
   - `watermark`: False (不添加水印)
3. 输入prompt，执行节点
4. 节点将生成指定数量的图片

## ⚙️ 工作原理

### 架构设计
采用**职责单一原则**，按并发场景拆分节点：
- **Banana2Batch**: 专门处理Banana系列模型的批量并发
- **DoubaoBatch**: 专门处理豆包模型的批量并发
- **ModelCompare**: 专门处理单任务多模型比较

### 双供应商支持

#### Comfly供应商
- **支持模型**: nano-banana系列、doubao-seedream系列
- **API格式**: OpenAI DALL-E兼容格式
- **异步模式**: 所有模型都使用异步task_id轮询
- **并发控制**: 最大5个并发请求

#### BananaWebAPI供应商
- **支持模型**: nano-banana系列
- **API格式**: 专用NanoBanana格式
- **异步模式**: 使用task_id轮询
- **图片上传**: 支持OSS自动上传

### 任务执行流程
1. **输入解析**: 根据节点类型和供应商解析对应的输入参数
2. **并发执行**: 使用asyncio实现不同粒度的并发控制
3. **API适配**: 自动选择对应供应商的API格式和参数
4. **结果合并**: 按节点类型返回对应的输出格式

### 异步模式说明

#### Comfly供应商 (所有模型)
- **触发条件**: 自动对所有Comfly模型启用异步模式
- **执行方式**: async=true参数 + task_id轮询
- **查询间隔**: 每5秒检查一次状态
- **超时控制**: 最长等待用户设置的timeout时间

#### BananaWebAPI供应商
- **执行方式**: task_id异步轮询模式
- **API调用**: POST /api/v1/nanobanana/generate-pro
- **状态查询**: GET /api/v1/nanobanana/record-info (每5秒)
- **图片处理**: 支持Base64和OSS URL上传

### 有效任务判断
- **Text2Img模式**: 只需要有效的prompt
- **Img2Img模式**: 需要有效的图像和prompt
- 空值判断：图像为None/空tensor，文本为空字符串

### API适配
- **Comfly文生图**: POST /v1/images/generations?async=true (JSON格式)
- **Comfly图生图**: POST /v1/images/edits?async=true (Multipart格式)
- **BananaWebAPI**: POST /api/v1/nanobanana/generate-pro (JSON格式)
- **状态查询**: 自动适配不同供应商的查询接口

## 🔧 技术实现

### 依赖要求
- Python 3.8+
- torch>=1.12.0 (ComfyUI自带)
- numpy>=1.21.0
- Pillow>=9.0.0
- requests>=2.25.0
- oss2 (可选，用于OSS上传)

### 架构特点
- **双供应商支持**: 自动适配Comfly和BananaWebAPI
- **模块化设计**: 可扩展支持更多API供应商和模型
- **异步处理**: 不阻塞ComfyUI主线程
- **内存优化**: 合理的图像处理和内存管理
- **错误隔离**: 单任务失败不影响其他任务
- **OSS集成**: 支持阿里云OSS自动图片上传

## 🚨 注意事项

### 安全提醒
- API密钥请妥善保管，避免明文存储
- 网络请求涉及安全风险，注意防火墙设置
- 大文件上传可能消耗较多带宽

### 性能考虑
- 单次请求超时建议不超过300秒
- 并发数量限制为5，避免服务过载
- 大批量任务建议分批执行

### 故障排除
- **节点未显示**: 检查__init__.py和节点注册，完全重启ComfyUI
- **API调用失败**: 确认API密钥、网络连接和供应商选择
- **供应商切换**: 更换供应商时需要重新配置对应的base_url和model
- **OSS上传失败**: 检查OSS配置参数和网络连接
- **图像格式错误**: 确保输入图像为有效PNG/JPEG格式
- **内存不足**: 减少并发数量或输入图像分辨率
- **异步超时**: 适当增加timeout参数值

## 📝 更新日志

### v2.0.0 (最新)
- ✅ 双供应商支持: Comfly + BananaWebAPI
- ✅ 自动API适配: 根据供应商选择正确的API格式
- ✅ OSS图片上传集成: 支持阿里云OSS自动上传
- ✅ 异步模式优化: 所有Comfly模型使用异步处理
- ✅ 模型比较功能: 单任务多模型并发比较
- ✅ 错误处理完善: 详细的状态反馈和重试机制
- ✅ 模块化重构: 按场景拆分专门的节点

### v1.0.0
- ✅ 基础框架搭建
- ✅ Comfly API集成 (nano-banana和豆包模型)
- ✅ 文生图、图生图、多图生图支持
- ✅ 并发异步处理
- ✅ 重试机制
- ✅ 完善的输入输出系统

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 项目文件结构
```
AIYang_comfyui_myapi/
├── __init__.py                    # 节点注册和集成
├── banana2_batch_node.py          # Comfly/BananaWebAPI批量并发节点
├── doubao_batch_node.py           # 豆包批量并发节点
├── model_compare_node.py          # 模型比较节点
├── requirements.txt               # 依赖说明
├── README.md                      # 详细使用文档

```

### 扩展开发
- **添加新供应商**: 在`_build_api_request`中添加新的供应商判断逻辑
- **添加新模型**: 在对应节点的INPUT_TYPES中扩展模型选项
- **自定义API适配**: 参考现有的Comfly和BananaWebAPI实现
- **优化性能**: 改进并发控制和内存管理
- **OSS扩展**: 支持更多云存储服务集成

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。
