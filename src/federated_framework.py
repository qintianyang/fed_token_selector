import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any # 导入 Any
import logging
import copy
from abc import ABC, abstractmethod

from token_selector import TokenSelectorController
from llm_interface import LLMInterfaceFactory, BaseLLMInterface
from dataset import WatermarkDataset, collate_fn # 从新模块导入

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    """
    联邦学习客户端
    """
    def __init__(self, 
                 client_id: int, 
                 model: nn.Module, 
                 data_path: str, # 接收数据路径
                 config: Dict, 
                 device: str = 'cpu'):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.config = config
        self.device = device
        self.llm_interface = None

        # 动态创建LLM接口
        llm_config = self.config.get('llm_config', {})
        if llm_config.get('enabled', False):
            try:
                self.llm_interface = LLMInterfaceFactory.create_llm_interface(
                    interface_type=llm_config['type'],
                    **llm_config.get(llm_config['type'], {})
                )
            except Exception as e:
                logger.error(f"客户端 {self.client_id} 创建LLM接口失败: {e}")

        # 使用data_path创建数据集和数据加载器
        self.dataset = WatermarkDataset(
            config=self.config,
            data_path=data_path,
            llm_interface=self.llm_interface
        )
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collate_fn
        )

        # 确保learning_rate是浮点数类型
        learning_rate = self.config.get('learning_rate', 0.001)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.training_stats: Dict[str, List[float]] = {
            'total_loss': [],
            'watermark_loss': [],
            'semantic_loss': [],
            'fluency_loss': []
        }

    def local_train(self, num_epochs: int) -> Dict[str, List[float]]:
        """执行本地训练并返回统计信息"""
        self.model.train()

        for epoch in range(num_epochs):
            epoch_losses = {key: [] for key in self.training_stats.keys()}
            batch_id = 0
            for batch in self.data_loader:
                batch_id += 1
                print(f"客户端 {self.client_id} epoch {epoch}正在处理批次 {batch_id}")
                self.optimizer.zero_grad()

                # 从LLM获取数据（如果可用）
                batch = self._get_llm_data_for_batch(batch)

                context_tokens = batch['context_tokens'].to(self.device)
                top_k_logits = batch['top_k_logits'].to(self.device)
                top_k_indices = batch['top_k_indices'].to(self.device)
                target_bit = batch['target_bit'].to(self.device).squeeze(-1)

                # 模型前向传播
                selection_logits, _ = self.model(
                    context_embedding=context_tokens, # 修复：传递token IDs
                    top_k_logits=top_k_logits,
                    target_bit=target_bit.long(),
                    top_k_indices=top_k_indices
                )

                # 从 logits 计算 probs，用于需要概率的损失函数
                selection_probs = F.softmax(selection_logits, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)

                # 使用 top-p 采样来确定 green_mask
                top_p = self.config.get('top_p', 0.9)
                sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率高于top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 至少保留一个token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 创建green_mask
                green_mask = torch.ones_like(top_k_logits, dtype=torch.bool)
                
                # 使用相对索引来更新green_mask
                rows = torch.arange(green_mask.size(0)).unsqueeze(1)
                green_mask[rows, sorted_indices[sorted_indices_to_remove]] = False

                watermark_loss = self.model.get_watermark_loss(selection_logits, target_bit, green_mask)
                semantic_loss = self.model.get_semantic_loss(selection_probs, top_k_logits)
                fluency_loss = self.model.get_fluency_loss(selection_probs, top_k_probs, green_mask)

                # 计算损失
                # 假设 loss_components 是一个包含不同损失项的字典
                # total_loss = loss_components['total_loss']

                total_loss = watermark_loss + semantic_loss + fluency_loss
                total_loss.backward()
                self.optimizer.step()

                epoch_losses['total_loss'].append(total_loss.item())
                epoch_losses['watermark_loss'].append(watermark_loss.item())
                epoch_losses['semantic_loss'].append(semantic_loss.item())
                epoch_losses['fluency_loss'].append(fluency_loss.item())

            # 记录并打印每个epoch的平均损失
            for loss_name, loss_values in epoch_losses.items():
                if loss_values:
                    avg_loss = np.mean(loss_values)
                    self.training_stats[loss_name].append(avg_loss)
                    print(f"客户端 {self.client_id} Epoch {epoch}: Avg {loss_name} = {avg_loss:.4f}")

        # 返回训练统计信息
        return self.training_stats

    def _get_llm_data_for_batch(self, batch: Dict) -> Dict:
        """如果LLM可用，则使用LLM增强数据批次"""
        if self.llm_interface is None:
            return batch

        try:
            # 假设batch中已有context_tokens
            context_tokens_list = batch['context_tokens'].tolist()
            new_logits = []
            new_indices = []

            for tokens in context_tokens_list:
                logits = self.llm_interface.get_next_token_logits(tokens)
                # 此处需要添加从logits到top_k_logits和top_k_indices的转换
                # 为简单起见，我们暂时跳过这一步，并返回原始批次
                # 在实际实现中，您需要像WatermarkDataset中那样实现核采样
                pass # Placeholder

            # 如果成功获取了新的logits和indices，则更新批次
            # batch['top_k_logits'] = torch.stack(new_logits)
            # batch['top_k_indices'] = torch.stack(new_indices)

        except Exception as e:
            logger.warning(f"客户端 {self.client_id} 从LLM获取数据失败: {e}，使用模拟数据")

        return batch

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """获取当前模型状态"""
        return self.model.state_dict()
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return self.training_stats


class FederatedAggregator(ABC):
    """
    联邦学习聚合器基类
    """
    @abstractmethod
    def aggregate(self, 
                  server_model: nn.Module, 
                  client_updates: List[Dict[str, torch.Tensor]], 
                  client_weights: List[float]) -> nn.Module:
        pass

class FedAvgAggregator(FederatedAggregator):
    """FedAvg聚合算法"""
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg聚合：加权平均
        
        Args:
            client_updates: 客户端模型更新列表
            client_weights: 客户端权重（通常基于数据量）
            
        Returns:
            aggregated_update: 聚合后的模型更新
        """
        if not client_updates:
            return {}
        
        # 归一化权重
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # 初始化聚合更新
        aggregated_update = {}
        for key in client_updates[0].keys():
            aggregated_update[key] = torch.zeros_like(client_updates[0][key])
        
        # 加权聚合
        for client_update, weight in zip(client_updates, normalized_weights):
            for key in aggregated_update:
                aggregated_update[key] += weight * client_update[key]
        
        return aggregated_update


class FedProxAggregator(FederatedAggregator):
    """FedProx聚合算法（带正则化）"""
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # 正则化参数
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedProx聚合：加权平均 + 正则化
        """
        # 先使用FedAvg
        fedavg_aggregator = FedAvgAggregator()
        aggregated_update = fedavg_aggregator.aggregate(client_updates, client_weights)
        
        # 应用正则化（简化版本）
        for key in aggregated_update:
            aggregated_update[key] *= (1 - self.mu)
        
        return aggregated_update


class FederatedServer:
    """
    联邦学习服务器
    
    负责：
    1. 管理全局模型
    2. 协调客户端训练
    3. 聚合模型更新
    4. 评估模型性能
    """
    
    def __init__(
        self,
        global_model: TokenSelectorController,
        aggregator: FederatedAggregator,
        config: Dict[str, Any]
    ):
        self.global_model = global_model
        self.aggregator = aggregator
        self.config = config
        
        # 训练配置
        self.num_rounds = config.get('num_rounds', 100)
        self.min_clients = config.get('min_clients', 2)
        self.client_fraction = config.get('client_fraction', 1.0)
        
        # 日志
        self.logger = logging.getLogger('FederatedServer')
        
        # 训练历史
        self.training_history = {
            'round': [],
            'num_clients': [],
            'avg_loss': [],
            'model_norm': []
        }
    
    def train(
        self,
        clients: List[FederatedClient],
        validation_data: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        联邦训练主循环
        
        Args:
            clients: 客户端列表
            validation_data: 验证数据（可选）
            
        Returns:
            training_results: 训练结果
        """
        self.logger.info(f'开始联邦训练，共 {self.num_rounds} 轮，{len(clients)} 个客户端')
        
        for round_num in range(self.num_rounds):
            round_start_time = time.time()
            
            # 选择参与训练的客户端
            selected_clients = self._select_clients(clients)
            self.logger.info(f'第 {round_num + 1} 轮：选择了 {len(selected_clients)} 个客户端')
            
            # 客户端本地训练
            client_updates = []
            client_weights = []
            
            global_state = self.global_model.state_dict()
            
            for client in selected_clients:
                self.logger.debug(f'客户端 {client.client_id} 开始本地训练')
                
                # 本地训练
                model_update = client.local_train(global_state)
                client_updates.append(model_update)
                
                # 客户端权重（基于数据量）
                client_weight = len(client.local_data.dataset)
                client_weights.append(client_weight)
                
                self.logger.debug(f'客户端 {client.client_id} 完成本地训练')
            
            # 聚合模型更新
            aggregated_update = self.aggregator.aggregate(client_updates, client_weights)
            
            # 更新全局模型
            self._update_global_model(aggregated_update)
            
            # 评估模型
            round_metrics = self._evaluate_round(selected_clients, validation_data)
            
            # 记录训练历史
            self.training_history['round'].append(round_num + 1)
            self.training_history['num_clients'].append(len(selected_clients))
            self.training_history['avg_loss'].append(round_metrics.get('avg_loss', 0.0))
            self.training_history['model_norm'].append(round_metrics.get('model_norm', 0.0))
            
            round_time = time.time() - round_start_time
            self.logger.info(
                f'第 {round_num + 1} 轮完成，耗时 {round_time:.2f}s，'
                f'平均损失: {round_metrics.get("avg_loss", 0.0):.4f}'
            )
        
        self.logger.info('联邦训练完成')
        
        return {
            'global_model': self.global_model,
            'training_history': self.training_history,
            'final_metrics': round_metrics
        }
    
    def _select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        """选择参与训练的客户端"""
        num_selected = max(self.min_clients, int(len(clients) * self.client_fraction))
        num_selected = min(num_selected, len(clients))
        
        # 随机选择
        selected_indices = np.random.choice(len(clients), num_selected, replace=False)
        return [clients[i] for i in selected_indices]
    
    def _update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """更新全局模型"""
        current_state = self.global_model.state_dict()
        
        for key in current_state:
            if key in aggregated_update:
                current_state[key] += aggregated_update[key]
        
        self.global_model.load_state_dict(current_state)
    
    def _evaluate_round(
        self,
        selected_clients: List[FederatedClient],
        validation_data: Optional[DataLoader]
    ) -> Dict[str, float]:
        """评估当前轮次"""
        metrics = {}
        
        # 计算平均损失
        total_losses = []
        for client in selected_clients:
            client_stats = client.get_training_stats()
            if client_stats['total_loss']:
                total_losses.extend(client_stats['total_loss'][-client.local_epochs:])
        
        if total_losses:
            metrics['avg_loss'] = np.mean(total_losses)
        
        # 计算模型范数
        model_norm = 0.0
        for param in self.global_model.parameters():
            model_norm += param.data.norm(2).item() ** 2
        metrics['model_norm'] = model_norm ** 0.5
        
        # 验证集评估（如果提供）
        if validation_data is not None:
            val_metrics = self._validate(validation_data)
            metrics.update(val_metrics)
        
        return metrics
    
    def _validate(self, validation_data: DataLoader) -> Dict[str, float]:
        """在验证集上评估模型"""
        self.global_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        bit_generator = WatermarkBitGenerator(seed=999)  # 固定种子用于验证
        greenlist_generator = GreenListGenerator(
            vocab_size=self.config.get('vocab_size', 50000),
            gamma=self.config.get('gamma', 0.25)
        )
        
        with torch.no_grad():
            for batch in validation_data:
                context_embedding = batch['context_embedding']
                top_k_logits = batch['top_k_logits']
                top_k_indices = batch['top_k_indices']
                context_tokens = batch['context_tokens']
                
                batch_size = context_embedding.size(0)
                
                # 生成目标比特
                target_bits = bit_generator.generate_bits(batch_size)
                target_bit_tensor = torch.tensor(target_bits, dtype=torch.long)
                
                # 生成绿名单
                green_mask = greenlist_generator.get_greenlist_mask(context_tokens, top_k_indices)
                
                # 前向传播
                selection_probs, _ = self.global_model(
                    context_embedding, top_k_logits, target_bit_tensor, top_k_indices
                )
                
                # 计算损失
                watermark_loss = self.global_model.get_watermark_loss(
                    selection_probs, target_bit_tensor, green_mask
                )
                semantic_loss = self.global_model.get_semantic_loss(selection_probs, top_k_logits)
                fluency_loss = self.global_model.get_fluency_loss(selection_probs, top_k_logits)
                
                total_loss += (watermark_loss + semantic_loss + fluency_loss).item()
                num_batches += 1
        
        self.global_model.train()
        
        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def save_model(self, path: str):
        """保存全局模型"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        self.logger.info(f'模型已保存到 {path}')
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.logger.info(f'模型已从 {path} 加载')