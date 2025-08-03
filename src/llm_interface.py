import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import logging
from abc import ABC, abstractmethod
import json
import time
import requests
from dataclasses import dataclass

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """大模型响应数据结构"""
    logits: torch.Tensor  # [vocab_size] 或 [seq_len, vocab_size]
    tokens: List[int]     # 对应的token序列
    text: str            # 生成的文本
    metadata: Dict       # 额外的元数据

class BaseLLMInterface(ABC):
    """大语言模型接口抽象基类"""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        
    @abstractmethod
    def get_next_token_logits(self, 
                             context_tokens: List[int], 
                             **kwargs) -> torch.Tensor:
        """获取下一个token的logits分布"""
        pass
    
    @abstractmethod
    def generate_text(self, 
                     prompt: str, 
                     max_length: int = 50,
                     **kwargs) -> LLMResponse:
        """生成文本"""
        pass
    
    def get_top_k_logits(self, 
                        context_tokens: List[int], 
                        k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取top-k的logits和对应的token索引"""
        logits = self.get_next_token_logits(context_tokens)
        top_k_logits, top_k_indices = torch.topk(logits, k)
        return top_k_logits, top_k_indices

class OpenAIInterface(BaseLLMInterface):
    """OpenAI API接口"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", config: Dict = None):
        super().__init__(model_name, config or {})
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 注意：OpenAI API通常不直接提供logits，这里需要特殊处理
        logger.warning("OpenAI API不直接提供logits，将使用近似方法")
    
    def get_next_token_logits(self, context_tokens: List[int], **kwargs) -> torch.Tensor:
        """通过多次采样近似获取logits分布"""
        # 将token转换为文本
        context_text = self._tokens_to_text(context_tokens)
        
        # 使用多次采样来估计概率分布
        token_counts = {}
        num_samples = kwargs.get('num_samples', 100)
        
        for _ in range(num_samples):
            try:
                response = self._call_api(context_text, max_tokens=1, temperature=1.0)
                if response and 'choices' in response:
                    next_token = response['choices'][0]['message']['content']
                    token_id = self._text_to_token(next_token)
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
            except Exception as e:
                logger.warning(f"API调用失败: {e}")
                continue
        
        # 转换为logits
        logits = torch.full((self.vocab_size,), -10.0)  # 默认很低的logits
        for token_id, count in token_counts.items():
            if 0 <= token_id < self.vocab_size:
                logits[token_id] = np.log(count / num_samples)
        
        return logits
    
    def generate_text(self, prompt: str, max_length: int = 50, **kwargs) -> LLMResponse:
        """生成文本"""
        try:
            response = self._call_api(prompt, max_tokens=max_length, **kwargs)
            if response and 'choices' in response:
                generated_text = response['choices'][0]['message']['content']
                tokens = self._text_to_tokens(generated_text)
                
                # 创建模拟的logits（实际应用中需要更复杂的处理）
                logits = torch.randn(len(tokens), self.vocab_size)
                
                return LLMResponse(
                    logits=logits,
                    tokens=tokens,
                    text=generated_text,
                    metadata={'model': self.model_name, 'response': response}
                )
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return self._create_fallback_response(prompt)
    
    def _call_api(self, prompt: str, **kwargs) -> Dict:
        """调用OpenAI API"""
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 50),
            'temperature': kwargs.get('temperature', 0.7)
        }
        
        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers=self.headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code}, {response.text}")
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """将token序列转换为文本（需要tokenizer）"""
        # 简化实现，实际需要使用对应的tokenizer
        return f"<tokens:{','.join(map(str, tokens))}>"
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """将文本转换为token序列（需要tokenizer）"""
        # 简化实现，实际需要使用对应的tokenizer
        return [hash(char) % self.vocab_size for char in text[:50]]
    
    def _text_to_token(self, text: str) -> int:
        """将单个字符/词转换为token ID"""
        return hash(text) % self.vocab_size
    
    def _create_fallback_response(self, prompt: str) -> LLMResponse:
        """创建fallback响应"""
        fallback_text = "Error: Unable to generate text"
        tokens = self._text_to_tokens(fallback_text)
        logits = torch.randn(len(tokens), self.vocab_size)
        
        return LLMResponse(
            logits=logits,
            tokens=tokens,
            text=fallback_text,
            metadata={'error': True}
        )

class HuggingFaceInterface(BaseLLMInterface):
    """HuggingFace模型接口"""
    
    def __init__(self, model_name: str = "gpt2", config: Dict = None):
        super().__init__(model_name, config or {})
        
        # 处理'auto'设备选择
        device_config = config.get('device', 'cpu')
        if device_config == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_config
            
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载HuggingFace模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 更新词汇表大小
            self.vocab_size = self.tokenizer.vocab_size
            
            logger.info(f"成功加载模型: {self.model_name}")
            
        except ImportError:
            logger.error("请安装transformers库: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_next_token_logits(self, context_tokens: List[int], **kwargs) -> torch.Tensor:
        """获取下一个token的logits"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        with torch.no_grad():
            input_ids = torch.tensor([context_tokens]).to(self.device)
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # 取最后一个位置的logits
            
        return logits.cpu()
    
    def generate_text(self, prompt: str, max_length: int = 50, **kwargs) -> LLMResponse:
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型或tokenizer未加载")
        
        try:
            # 编码输入
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 生成参数
            generation_kwargs = {
                'max_length': input_ids.shape[1] + max_length,
                'temperature': kwargs.get('temperature', 0.7),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'return_dict_in_generate': True,
                'output_scores': True
            }
            
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **generation_kwargs)
            
            # 解析结果
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 获取logits
            if hasattr(outputs, 'scores') and outputs.scores:
                logits = torch.stack(outputs.scores, dim=0)  # [seq_len, vocab_size]
            else:
                logits = torch.randn(len(generated_ids), self.vocab_size)
            
            return LLMResponse(
                logits=logits.cpu(),
                tokens=generated_ids.cpu().tolist(),
                text=generated_text,
                metadata={'model': self.model_name}
            )
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return self._create_fallback_response(prompt)
    
    def _create_fallback_response(self, prompt: str) -> LLMResponse:
        """创建fallback响应"""
        fallback_text = "Error: Unable to generate text"
        tokens = [1, 2, 3]  # 简单的fallback tokens
        logits = torch.randn(len(tokens), self.vocab_size)
        
        return LLMResponse(
            logits=logits,
            tokens=tokens,
            text=fallback_text,
            metadata={'error': True}
        )

class LocalModelInterface(BaseLLMInterface):
    """本地模型接口（用于自定义模型）"""
    
    def __init__(self, model_path: str, config: Dict = None):
        super().__init__(model_path, config or {})
        self.model_path = model_path
        self.device = config.get('device', 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载本地模型"""
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"成功加载本地模型: {self.model_path}")
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}")
            raise
    
    def get_next_token_logits(self, context_tokens: List[int], **kwargs) -> torch.Tensor:
        """获取下一个token的logits"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        with torch.no_grad():
            input_ids = torch.tensor([context_tokens]).to(self.device)
            logits = self.model(input_ids)
            if logits.dim() == 3:  # [batch, seq, vocab]
                logits = logits[0, -1, :]  # 取最后一个位置
            
        return logits.cpu()
    
    def generate_text(self, prompt: str, max_length: int = 50, **kwargs) -> LLMResponse:
        """生成文本（简化实现）"""
        # 这里需要根据具体的本地模型实现
        # 简化版本，返回模拟数据
        tokens = list(range(max_length))
        logits = torch.randn(max_length, self.vocab_size)
        
        return LLMResponse(
            logits=logits,
            tokens=tokens,
            text=f"Generated text for: {prompt}",
            metadata={'model': 'local', 'path': self.model_path}
        )

class LLMInterfaceFactory:
    """大模型接口工厂类"""
    
    @staticmethod
    def create_interface(interface_type: str, **kwargs) -> BaseLLMInterface:
        """创建大模型接口
        
        Args:
            interface_type: 接口类型 ('openai', 'huggingface', 'local')
            **kwargs: 接口配置参数
        """
        if interface_type.lower() == 'openai':
            return OpenAIInterface(**kwargs)
        elif interface_type.lower() == 'huggingface':
            return HuggingFaceInterface(**kwargs)
        elif interface_type.lower() == 'local':
            return LocalModelInterface(**kwargs)
        else:
            raise ValueError(f"不支持的接口类型: {interface_type}")
    
    @staticmethod
    def get_available_interfaces() -> List[str]:
        """获取可用的接口类型"""
        return ['openai', 'huggingface', 'local']

# 使用示例
if __name__ == "__main__":
    # 测试HuggingFace接口
    try:
        config = {'device': 'cpu'}
        llm = LLMInterfaceFactory.create_interface('huggingface', model_name='meta-llama/Llama-3.2-1B', config=config)
        
        # 测试获取logits
        context = [1, 2, 3, 4, 5]
        logits = llm.get_next_token_logits(context)
        print(f"Logits shape: {logits.shape}")
        
        # 测试文本生成
        response = llm.generate_text("Hello, world!", max_length=20)
        print(f"Generated: {response.text}")
        
    except Exception as e:
        print(f"测试失败: {e}")