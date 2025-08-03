import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class TokenSelectorController(nn.Module):
    """
    Token选择器控制器 - 联邦学习的核心模型
    
    功能：
    - 输入：当前上下文、大模型logits、目标水印比特
    - 输出：选择的token概率分布
    - 目标：在保持语义的同时嵌入水印比特
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        top_k: int = 50
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Logits编码器
        self.logits_encoder = nn.Sequential(
            nn.Linear(top_k, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 比特嵌入
        self.bit_embedding = nn.Embedding(2, hidden_dim // 4)  # 0或1
        
        # 融合层
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, top_k),
            nn.Softmax(dim=-1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        context_embedding: torch.Tensor,  # [batch_size, context_dim]
        top_k_logits: torch.Tensor,      # [batch_size, top_k]
        target_bit: torch.Tensor,        # [batch_size] 0或1
        top_k_indices: torch.Tensor      # [batch_size, top_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            selection_probs: [batch_size, top_k] - 在top_k中的选择概率
            selected_indices: [batch_size] - 选择的token在原vocab中的索引
        """
        batch_size = context_embedding.size(0)
        
        # 编码各个输入
        context_encoded = self.context_encoder(context_embedding)  # [batch_size, hidden_dim]
        logits_encoded = self.logits_encoder(top_k_logits)         # [batch_size, hidden_dim]
        bit_encoded = self.bit_embedding(target_bit)               # [batch_size, hidden_dim//4]
        
        # 扩展bit编码维度
        bit_encoded = F.pad(bit_encoded, (0, self.hidden_dim - self.hidden_dim // 4))
        
        # 融合特征
        fused_features = context_encoded + logits_encoded + bit_encoded  # [batch_size, hidden_dim]
        fused_features = fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 通过Transformer层
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)
        
        # 输出选择概率
        fused_features = fused_features.squeeze(1)  # [batch_size, hidden_dim]
        selection_probs = self.output_projection(fused_features)  # [batch_size, top_k]
        
        # 选择token
        selected_top_k_idx = torch.multinomial(selection_probs, 1).squeeze(-1)  # [batch_size]
        selected_indices = top_k_indices.gather(1, selected_top_k_idx.unsqueeze(1)).squeeze(1)
        
        return selection_probs, selected_indices
    
    def get_watermark_loss(
        self,
        selection_probs: torch.Tensor,
        target_bit: torch.Tensor,
        green_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算水印损失 - 鼓励选择绿名单token当target_bit=1
        
        Args:
            selection_probs: [batch_size, top_k]
            target_bit: [batch_size]
            green_mask: [batch_size, top_k] - 绿名单mask
        """
        # 绿名单概率
        green_probs = (selection_probs * green_mask).sum(dim=-1)  # [batch_size]
        
        # 当target_bit=1时，希望green_probs高；target_bit=0时，希望green_probs低
        target_green_probs = target_bit.float()  # 转换为浮点数
        
        # 使用BCE损失
        watermark_loss = F.binary_cross_entropy(green_probs, target_green_probs)
        
        return watermark_loss
    
    def get_semantic_loss(
        self,
        selection_probs: torch.Tensor,
        original_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算语义保持损失 - 与原始模型输出的KL散度
        
        Args:
            selection_probs: [batch_size, top_k]
            original_logits: [batch_size, top_k]
        """
        original_probs = F.softmax(original_logits, dim=-1)
        semantic_loss = F.kl_div(
            F.log_softmax(selection_probs, dim=-1),
            original_probs,
            reduction='batchmean'
        )
        return semantic_loss
    
    def get_fluency_loss(
        self,
        selection_probs: torch.Tensor,
        top_k_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算流畅性损失 - 鼓励选择高概率token
        
        Args:
            selection_probs: [batch_size, top_k]
            top_k_logits: [batch_size, top_k]
        """
        # 原始概率分布
        original_probs = F.softmax(top_k_logits, dim=-1)
        
        # 期望的原始概率（加权平均）
        expected_original_prob = (selection_probs * original_probs).sum(dim=-1)
        
        # 鼓励选择高概率token
        fluency_loss = -expected_original_prob.mean()
        
        return fluency_loss


class WatermarkBitGenerator:
    """
    水印比特生成器 - 生成伪随机的水印比特序列
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.bit_stream = []
        self.position = 0
    
    def generate_bits(self, length: int) -> List[int]:
        """生成指定长度的比特序列"""
        if len(self.bit_stream) < self.position + length:
            # 扩展比特流
            new_bits = self.rng.randint(0, 2, size=length * 2).tolist()
            self.bit_stream.extend(new_bits)
        
        bits = self.bit_stream[self.position:self.position + length]
        self.position += length
        return bits
    
    def reset(self):
        """重置位置"""
        self.position = 0


class GreenListGenerator:
    """
    绿名单生成器 - 基于上下文生成绿名单token
    """
    
    def __init__(self, vocab_size: int, gamma: float = 0.25, hash_key: int = 15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key
        self.greenlist_size = int(vocab_size * gamma)
    
    def get_greenlist_mask(
        self,
        context_tokens: torch.Tensor,  # [batch_size, seq_len]
        top_k_indices: torch.Tensor    # [batch_size, top_k]
    ) -> torch.Tensor:
        """
        生成绿名单mask
        
        Returns:
            green_mask: [batch_size, top_k] - 1表示绿名单token，0表示非绿名单
        """
        batch_size, top_k = top_k_indices.shape
        green_mask = torch.zeros_like(top_k_indices, dtype=torch.float)
        
        for b_idx in range(batch_size):
            # 基于上下文最后一个token生成种子
            if context_tokens.size(1) > 0:
                seed = self.hash_key * context_tokens[b_idx, -1].item()
            else:
                seed = self.hash_key
            
            # 生成绿名单
            rng = torch.Generator()
            rng.manual_seed(seed)
            vocab_permutation = torch.randperm(self.vocab_size, generator=rng)
            greenlist_ids = vocab_permutation[:self.greenlist_size]
            
            # 检查top_k中哪些是绿名单
            for k_idx, token_id in enumerate(top_k_indices[b_idx]):
                if token_id.item() in greenlist_ids:
                    green_mask[b_idx, k_idx] = 1.0
        
        return green_mask