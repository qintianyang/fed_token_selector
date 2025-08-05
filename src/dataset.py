import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from tqdm import tqdm

from llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)

class WatermarkDataset(Dataset):
    """
    水印训练数据集
    从真实文本数据加载，或生成模拟数据
    """
    
    def __init__(self, 
                 config: Dict,
                 data_path: Optional[str] = None,
                 llm_interface: Optional[BaseLLMInterface] = None):
        self.config = config
        self.data_path = data_path
        self.llm_interface = llm_interface
        
        # 从配置中获取参数
        self.vocab_size = config.get('vocab_size', 50000)
        self.max_seq_len = config.get('max_seq_len', 128)
        self.top_p = config.get('model', {}).get('top_p', 0.92)
        self.max_candidate_tokens = config.get('model', {}).get('max_candidate_tokens', 50)
        self.num_samples = config.get('samples_per_client', 1000)
        
        # 加载或生成数据
        if self.data_path and os.path.exists(self.data_path):
            self.data = self._load_real_data()
        else:
            logger.warning(f"数据文件未找到或未提供，将使用合成数据。")
            self.data = self._generate_synthetic_data()

    def _load_real_data(self) -> List[Dict]:
        """从真实数据文件加载和处理数据"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 根据需要对数据进行采样
        if self.num_samples is not None and len(lines) > self.num_samples:
            lines = lines[:self.num_samples]
            
        for line in tqdm(lines, desc="处理真实数据"):
            text = line.strip()
            if not text or self.llm_interface is None:
                continue

            try:
                # 将文本转换为token
                tokens = self.llm_interface.tokenizer.encode(text)
                
                # 将数据分割为prompt和completion
                if len(tokens) < 20: continue # 忽略太短的文本
                split_point = len(tokens) - 10 # 假设最后10个token为completion
                prompt_tokens = tokens[:split_point]
                
                # 获取LLM的logits
                logits = self.llm_interface.get_next_token_logits(prompt_tokens)
                top_p_logits, top_p_indices = self._nucleus_sampling(logits)
                
                target_bit = torch.randint(0, 2, (1,)).float()
                
                data.append({
                    'context_tokens': torch.tensor(prompt_tokens, dtype=torch.long),
                    'top_p_logits': top_p_logits,
                    'top_p_indices': top_p_indices,
                    'target_bit': target_bit
                })
            except Exception as e:
                logger.warning(f"处理行失败: {e}")
                
        return data
        
    def _generate_synthetic_data(self) -> List[Dict]:
        """生成合成训练数据"""
        data = []
        
        for i in range(self.num_samples):
            # 模拟上下文序列
            seq_len = np.random.randint(10, self.max_seq_len)
            context_tokens = torch.randint(0, self.vocab_size, (seq_len,))
            
            # 模拟大模型输出的logits (未归一化)
            logits = torch.randn(self.vocab_size) * 2.0
            
            # 获取top-p token和对应的logits
            top_p_logits, top_p_indices = self._nucleus_sampling(logits)
            
            # 随机生成目标水印比特
            target_bit = torch.randint(0, 2, (1,)).float()
            
            data.append({
                'context_tokens': context_tokens,
                'top_p_logits': top_p_logits,
                'top_p_indices': top_p_indices,
                'target_bit': target_bit
            })
            
        return data
    
    def _nucleus_sampling(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据logits执行Top-p (nucleus)采样
        """
        probabilities = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probabilities[indices_to_remove] = 0
        
        probabilities /= torch.sum(probabilities, dim=-1, keepdim=True)
        
        top_logits, top_indices = torch.topk(probabilities, self.max_candidate_tokens)
        
        return top_logits, top_indices

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """数据批处理函数"""
    # 找到最大序列长度
    max_len = max(item['context_tokens'].size(0) for item in batch)
    
    # 填充序列
    context_tokens = []
    for item in batch:
        tokens = item['context_tokens']
        if tokens.size(0) < max_len:
            # 使用0填充
            padded = torch.cat([tokens, torch.zeros(max_len - tokens.size(0), dtype=tokens.dtype)])
        else:
            padded = tokens
        context_tokens.append(padded)
    
    return {
        'context_tokens': torch.stack(context_tokens),
        'top_k_logits': torch.stack([item['top_p_logits'] for item in batch]),
        'top_k_indices': torch.stack([item['top_p_indices'] for item in batch]),
        'target_bit': torch.stack([item['target_bit'] for item in batch])
    }