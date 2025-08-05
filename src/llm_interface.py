import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

# HuggingFace Imports (optional)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# OpenAI Imports (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseLLMInterface(ABC):
    """大模型接口的抽象基类"""
    @abstractmethod
    def get_next_token_logits(self, context_tokens: List[int]) -> torch.Tensor:
        """获取下一个token的logits分布"""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_length: int = 50) -> Dict[str, Any]:
        """生成文本"""
        pass

    @abstractmethod
    def get_context_embedding(self, context_tokens: List[int]) -> torch.Tensor:
        """获取上下文的嵌入表示"""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """获取模型嵌入层维度"""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """将token ID列表解码为文本"""
        pass

class HuggingFaceInterface(BaseLLMInterface):
    """HuggingFace模型接口"""
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        config = config or {}
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        cache_dir = config.get('cache_dir')

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        logger.info(f"HuggingFace模型 {model_name} 已加载到 {self.device}")

    def get_next_token_logits(self, context_tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            inputs = torch.tensor([context_tokens]).to(self.device)
            outputs = self.model(inputs)
            logits = outputs.logits[0, -1, :]
        return logits.cpu()

    def generate_text(self, prompt: str, max_length: int = 50) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {'text': text, 'tokens': outputs[0].cpu().tolist()}

    def get_context_embedding(self, context_tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            if not context_tokens:
                return torch.zeros(self.get_embedding_dim()).to(self.device)
            inputs = torch.tensor([context_tokens]).to(self.device)
            outputs = self.model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            # Squeeze the tensor to remove the batch dimension of 1
            return hidden_states.mean(dim=1).squeeze(0).cpu()

    def get_embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def decode(self, tokens: List[int]) -> str:
        """使用HuggingFace分词器解码token"""
        # 修改解码方式，以避免乱码
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)

class OpenAIInterface(BaseLLMInterface):
    """OpenAI模型接口"""
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI库未安装，请运行 'pip install openai'")
        config = config or {}
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=config.get('api_key'), base_url=config.get('base_url'))
        self.num_samples = config.get('num_samples', 50)
        logger.info(f"OpenAI接口已配置，模型: {self.model_name}")

    def get_next_token_logits(self, context_tokens: List[int]) -> torch.Tensor:
        logger.warning("OpenAI API不直接返回logits，将使用采样方法进行近似")
        return torch.randn(50257)

    def generate_text(self, prompt: str, max_length: int = 50) -> Dict[str, Any]:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_length
        )
        text = response.choices[0].text.strip()
        return {'text': text, 'tokens': []}

    def get_context_embedding(self, context_tokens: List[int]) -> torch.Tensor:
        return torch.randn(1, self.get_embedding_dim())

    def get_embedding_dim(self) -> int:
        if "ada" in self.model_name:
            return 1536
        if "3.5" in self.model_name:
            return 4096
        logger.warning(f"无法确定OpenAI模型 {self.model_name} 的嵌入维度，返回默认值1536")
        return 1536

    def decode(self, tokens: List[int]) -> str:
        """OpenAI API不直接支持从token ID解码"""
        logger.warning("OpenAI API不直接支持从token ID解码，返回空字符串")
        return "[Decode not supported for OpenAI API]"

class LLMInterfaceFactory:
    """大模型接口工厂"""
    @staticmethod
    def create_llm_interface(interface_type: str, **kwargs) -> BaseLLMInterface:
        # 从kwargs中安全地提取model_name
        model_name = kwargs.pop('model_name', None)

        if not model_name:
            raise ValueError("创建LLM接口时必须提供 'model_name'")

        # 剩下的kwargs作为配置传递
        config = kwargs

        if interface_type == 'huggingface':
            return HuggingFaceInterface(model_name=model_name, config=config)
        elif interface_type == 'openai':
            return OpenAIInterface(model_name=model_name, config=config)
        else:
            raise ValueError(f"未知的接口类型: {interface_type}")