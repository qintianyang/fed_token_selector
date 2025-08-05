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
        
        # 上下文token嵌入层
        self.context_embedding = nn.Embedding(vocab_size, context_dim)
        
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
            nn.Linear(hidden_dim // 2, top_k)
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
        context_embedding: torch.Tensor,  # [batch_size, context_dim] or [batch_size, seq_len] for token IDs
        top_k_logits: torch.Tensor,      # [batch_size, top_k]
        target_bit: torch.Tensor,        # [batch_size] 0或1
        top_k_indices: torch.Tensor      # [batch_size, top_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            context_embedding: 上下文嵌入 [batch_size, context_dim] 或 token IDs [batch_size, seq_len]
            top_k_logits: top-k logits [batch_size, top_k]
            target_bit: 目标比特 [batch_size]
            top_k_indices: top-k token索引 [batch_size, top_k]
            
        Returns:
            selection_probs: [batch_size, top_k] - 在top_k中的选择概率
            selected_indices: [batch_size] - 选择的token在原vocab中的索引
        """
        batch_size = context_embedding.size(0)
        
        # 处理上下文输入：如果是token IDs则转换为嵌入
        if context_embedding.dtype == torch.long:
            # 输入是token IDs，需要转换为嵌入
            if context_embedding.dim() == 2:  # [batch_size, seq_len]
                # 对序列进行平均池化得到固定维度的表示
                context_emb = self.context_embedding(context_embedding)  # [batch_size, seq_len, context_dim]
                context_embedding = context_emb.mean(dim=1)  # [batch_size, context_dim]
            else:  # [batch_size]
                context_embedding = self.context_embedding(context_embedding)  # [batch_size, context_dim]
        
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
        selection_logits = self.output_projection(fused_features)  # [batch_size, top_k]
        
        # 根据logits进行采样，得到在top_k中的索引
        selection_probs = F.softmax(selection_logits, dim=-1)
        selected_top_k_indices = torch.multinomial(selection_probs, 1).squeeze(-1) # [batch_size]

        # 从top_k_indices中获取在词汇表中的真实索引
        selected_indices = top_k_indices.gather(1, selected_top_k_indices.unsqueeze(-1)).squeeze(-1)

        # 注意：这里返回logits，而不是probs
        return selection_logits, selected_indices
    
    def get_watermark_loss(
        self,
        selection_logits: torch.Tensor, # 现在接收logits
        target_bit: torch.Tensor,
        green_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算水印损失 - 鼓励选择绿名单token当target_bit=1
        
        Args:
            selection_logits: [batch_size, top_k] - 未经激活的logits
            target_bit: [batch_size]
            green_mask: [batch_size, top_k] - 绿名单mask
        """
        # 我们需要计算“是绿色”这个事件的logit
        # 这不是一个简单的线性操作，因为softmax不是线性的
        # 一个近似方法是计算加权的logits，但这在数学上不完全正确
        # 更简单和直接的方法是，我们把问题看作是对每个token是否应该被选择的二元分类
        
        # 为了使用BCEWithLogitsLoss，我们需要一个logit来表示“选择绿色”的概率
        # 我们可以通过对绿区的logits进行聚合来得到这个值
        # 例如，使用log-sum-exp技巧来稳定地计算softmax后的概率的log值
        green_logits = selection_logits + (1.0 - green_mask.float()) * -1e9 # 用一个大负数mask掉非绿色的logits
        green_prob_logit = torch.logsumexp(green_logits, dim=-1) # [batch_size]

        target_green_probs = target_bit.float()  # 转换为浮点数
        
        # 使用BCEWithLogitsLoss
        watermark_loss = F.binary_cross_entropy_with_logits(green_prob_logit, target_green_probs)
        
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
    
    def get_fluency_loss(self, selection_probs, top_k_probs, green_mask):
        """
        计算流畅性损失，旨在惩罚模型选择非原始高概率词元。
        """
        # 只在绿区token上计算期望概率
        masked_selection_probs = selection_probs * green_mask
        # 避免除以零
        sum_selection_probs = masked_selection_probs.sum(dim=1, keepdim=True)
        normalized_selection_probs = masked_selection_probs / (sum_selection_probs + 1e-10)

        # 计算原始概率在绿区token上的期望值
        expected_original_prob = (normalized_selection_probs * top_k_probs).sum(dim=1)
        
        # 流畅性损失：我们希望最大化这个期望，所以最小化它的负值
        fluency_loss = 1 - expected_original_prob.mean()
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
    
    def message_to_bits(self, message: str) -> List[int]:
        """将字符串消息转换为比特序列"""
        # 将字符串转换为字节，再转换为比特
        message_bytes = message.encode('utf-8')
        bits = []
        for byte in message_bytes:
            # 将每个字节转换为8个比特
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def bits_to_message(self, bits: List[int]) -> str:
        """将比特序列转换回字符串消息"""
        # 确保比特数量是8的倍数
        if len(bits) % 8 != 0:
            # 填充到8的倍数
            bits = bits + [0] * (8 - len(bits) % 8)
        
        message_bytes = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << (7 - j))
            message_bytes.append(byte_value)
        
        try:
            return bytes(message_bytes).decode('utf-8')
        except UnicodeDecodeError:
            # 如果解码失败，返回十六进制表示
            return bytes(message_bytes).hex()
    
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
        # 确保green_mask在与输入张量相同的设备上
        device = top_k_indices.device
        green_mask = torch.zeros_like(top_k_indices, dtype=torch.float, device=device)
        
        for b_idx in range(batch_size):
            # 基于上下文最后一个token生成种子
            if context_tokens.size(1) > 0:
                seed = self.hash_key * context_tokens[b_idx, -1].item()
            else:
                seed = self.hash_key
            
            # 生成绿名单 - 确保在正确设备上
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            vocab_permutation = torch.randperm(self.vocab_size, generator=rng, device=device)
            greenlist_ids = vocab_permutation[:self.greenlist_size]
            
            # 检查top_k中哪些是绿名单
            for k_idx, token_id in enumerate(top_k_indices[b_idx]):
                if token_id.item() in greenlist_ids:
                    green_mask[b_idx, k_idx] = 1.0
        
        return green_mask