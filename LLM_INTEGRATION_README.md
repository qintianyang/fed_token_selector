# 大模型集成使用指南

本文档介绍如何在联邦学习Token选择器水印系统中集成和使用真实的大语言模型。

## 概述

原项目使用模拟的logits数据进行训练和测试。现在我们添加了真实大模型接口支持，可以：

1. **在训练阶段**：使用真实大模型的输出logits训练Token选择器
2. **在生成阶段**：使用真实大模型生成带水印的文本
3. **支持多种接口**：OpenAI API、HuggingFace模型、本地模型

## 新增文件

### 1. `src/llm_interface.py`
大模型接口模块，包含：
- `BaseLLMInterface`: 抽象基类
- `OpenAIInterface`: OpenAI API接口
- `HuggingFaceInterface`: HuggingFace模型接口
- `LocalModelInterface`: 本地模型接口
- `LLMInterfaceFactory`: 工厂类

### 2. `src/example_with_llm.py`
完整的使用示例，展示如何配置和使用大模型接口。

## 配置说明

### 配置文件 (`config/default_config.yaml`)

```yaml
llm_config:
  enabled: true  # 是否启用真实大模型接口
  type: "huggingface"  # 接口类型
  
  # HuggingFace配置
  huggingface:
    model_name: "gpt2"
    device: "auto"
    cache_dir: "./models"
  
  # OpenAI配置
  openai:
    model_name: "gpt-3.5-turbo"
    api_key: "your-api-key-here"
    base_url: "https://api.openai.com/v1"
    num_samples: 50
  
  # 本地模型配置
  local:
    model_path: "./models/local_model.pth"
    device: "auto"
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

新增的主要依赖：
- `transformers>=4.20.0`: HuggingFace模型支持
- `tokenizers>=0.12.0`: 分词器支持
- `requests>=2.28.0`: API调用支持

### 2. 基本使用

#### 创建大模型接口

```python
from src.llm_interface import LLMInterfaceFactory

# HuggingFace接口
hf_interface = LLMInterfaceFactory.create_interface(
    'huggingface',
    model_name='gpt2',
    config={'device': 'cpu'}
)

# OpenAI接口
openai_interface = LLMInterfaceFactory.create_interface(
    'openai',
    model_name='gpt-3.5-turbo',
    config={'api_key': 'your-api-key'}
)
```

#### 获取logits

```python
context_tokens = [1, 2, 3, 4, 5]
logits = hf_interface.get_next_token_logits(context_tokens)
print(f"Logits shape: {logits.shape}")  # [vocab_size]
```

#### 生成文本

```python
response = hf_interface.generate_text("Hello, world!", max_length=20)
print(f"Generated: {response.text}")
print(f"Tokens: {response.tokens}")
```

### 3. 联邦学习训练

```python
from src.train_federated import FederatedTrainer

config = {
    'vocab_size': 50257,
    'hidden_dim': 256,
    'top_k': 50,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'samples_per_client': 100,
    'num_rounds': 20,
    
    # 大模型配置
    'llm_config': {
        'enabled': True,
        'type': 'huggingface',
        'model_name': 'gpt2',
        'config': {'device': 'cpu'}
    }
}

trainer = FederatedTrainer(
    config=config,
    num_clients=3,
    device='cpu',
    llm_config=config['llm_config']
)

# 训练
for round_num in range(1, 6):
    metrics = trainer.train_round(round_num)
    print(f"Round {round_num}, Loss: {metrics['loss']:.4f}")
```

### 4. 水印文本生成

```python
from src.demo_complete import WatermarkTextGenerator

# 创建生成器
generator = WatermarkTextGenerator(
    token_selector=trained_model,
    vocab_size=50257,
    device='cpu',
    llm_config={
        'type': 'huggingface',
        'model_name': 'gpt2',
        'config': {'device': 'cpu'}
    }
)

# 生成带水印文本
prompt_tokens = [15496, 11, 995]  # "Hello, world"
generated_tokens, embedded_bits = generator.generate_text_with_watermark(
    prompt_tokens=prompt_tokens,
    max_length=50,
    watermark_message="SECRET"
)
```

## 接口类型详解

### 1. HuggingFace接口

**优点**：
- 支持大量开源模型
- 本地运行，无需API密钥
- 直接获取logits，精度高

**缺点**：
- 需要下载模型文件
- 计算资源要求较高

**配置示例**：
```python
config = {
    'type': 'huggingface',
    'model_name': 'gpt2',  # 或 'microsoft/DialoGPT-medium'
    'config': {
        'device': 'cuda',  # 或 'cpu'
        'cache_dir': './models'
    }
}
```

### 2. OpenAI接口

**优点**：
- 模型质量高
- 无需本地计算资源
- 支持最新模型

**缺点**：
- 需要API密钥和网络连接
- 不直接提供logits，需要近似估计
- 有使用成本

**配置示例**：
```python
config = {
    'type': 'openai',
    'model_name': 'gpt-3.5-turbo',
    'config': {
        'api_key': 'sk-...',
        'base_url': 'https://api.openai.com/v1',
        'num_samples': 100  # 用于估计概率分布
    }
}
```

### 3. 本地模型接口

**优点**：
- 完全自定义
- 无外部依赖
- 可针对特定任务优化

**缺点**：
- 需要自己训练或转换模型
- 实现复杂度较高

**配置示例**：
```python
config = {
    'type': 'local',
    'model_path': './models/my_model.pth',
    'config': {
        'device': 'cuda'
    }
}
```

## 运行示例

### 快速测试

```bash
cd src
python example_with_llm.py
```

### 完整演示

```bash
cd src
python demo_complete.py --mode full
```

## 性能优化建议

### 1. 设备选择
- 如果有GPU，设置 `device: 'cuda'`
- CPU运行时选择较小的模型（如gpt2而不是gpt2-large）

### 2. 批处理
- 适当增加batch_size以提高效率
- 但要注意内存限制

### 3. 模型选择
- 开发阶段使用小模型（gpt2）
- 生产环境可选择更大的模型

### 4. 缓存
- HuggingFace模型会自动缓存到本地
- 设置合适的cache_dir路径

## 故障排除

### 1. 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
- 减小batch_size
- 使用CPU：`device: 'cpu'`
- 选择更小的模型

### 2. 模型下载失败
```
OSError: Can't load tokenizer for 'gpt2'
```
**解决方案**：
- 检查网络连接
- 设置代理（如果需要）
- 手动下载模型文件

### 3. API调用失败
```
Exception: API调用失败: 401
```
**解决方案**：
- 检查API密钥是否正确
- 确认账户余额充足
- 检查网络连接

### 4. 依赖库问题
```
ImportError: No module named 'transformers'
```
**解决方案**：
```bash
pip install transformers tokenizers requests
```

## 扩展开发

### 添加新的大模型接口

1. 继承 `BaseLLMInterface`
2. 实现必要的方法
3. 在 `LLMInterfaceFactory` 中注册

```python
class CustomInterface(BaseLLMInterface):
    def get_next_token_logits(self, context_tokens, **kwargs):
        # 实现获取logits的逻辑
        pass
    
    def generate_text(self, prompt, max_length=50, **kwargs):
        # 实现文本生成的逻辑
        pass
```

### 自定义损失函数

可以在 `FederatedClient` 中修改损失函数权重：

```python
config = {
    'lambda_watermark': 1.0,  # 水印损失权重
    'lambda_semantic': 0.5,   # 语义损失权重
    'lambda_fluency': 0.3     # 流畅性损失权重
}
```

## 注意事项

1. **API密钥安全**：不要将API密钥提交到版本控制系统
2. **计算资源**：大模型需要较多计算资源，建议在GPU上运行
3. **网络连接**：某些接口需要稳定的网络连接
4. **版本兼容性**：确保transformers库版本与模型兼容
5. **许可证**：注意模型的使用许可证要求

## 更新日志

- **v1.0**: 添加基础大模型接口支持
- **v1.1**: 支持HuggingFace、OpenAI、本地模型
- **v1.2**: 集成到联邦学习框架
- **v1.3**: 添加配置文件支持和示例代码