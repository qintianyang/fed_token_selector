import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging
import numpy as np
from abc import ABC, abstractmethod
import json
import time

from token_selector import TokenSelectorController, WatermarkBitGenerator, GreenListGenerator
from llm_interface import BaseLLMInterface, LLMInterfaceFactory


class FederatedClient:
    """
    联邦学习客户端
    
    负责：
    1. 本地数据训练
    2. 模型参数更新
    3. 与服务器通信
    """
    
    def __init__(
        self,
        client_id: int,
        model: TokenSelectorController,
        local_data: DataLoader,
        config: Dict[str, Any],
        llm_interface: Optional[BaseLLMInterface] = None
    ):
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.config = config
    
        # 日志
        self.logger = logging.getLogger(f'Client_{client_id}')
        
        # 大模型接口
        self.llm_interface = llm_interface
        if self.llm_interface is None:
            # 如果没有提供LLM接口，创建默认接口
            llm_config = config.get('llm_config', {})
            interface_type = llm_config.get('type', 'huggingface')
            try:
                self.llm_interface = LLMInterfaceFactory.create_interface(
                    interface_type, **llm_config
                )
                self.logger.info(f"客户端 {client_id} 使用 {interface_type} 大模型接口")
            except Exception as e:
                self.logger.warning(f"大模型接口创建失败: {e}，将使用模拟数据")
                self.llm_interface = None
        
        # 训练配置
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 1e-3)
        self.local_epochs = training_config.get('local_epochs', 2)
        self.lambda_watermark = config.get('lambda_watermark', 1.0)
        self.lambda_semantic = config.get('lambda_semantic', 0.5)
        self.lambda_fluency = config.get('lambda_fluency', 0.3)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 工具类
        self.bit_generator = WatermarkBitGenerator(seed=client_id * 1000)
        self.greenlist_generator = GreenListGenerator(
            vocab_size=config.get('vocab_size', 50000),
            gamma=config.get('gamma', 0.25)
        )
        
        # 训练统计
        self.training_stats = {
            'total_loss': [],
            'watermark_loss': [],
            'semantic_loss': [],
            'fluency_loss': []
        }
    
    def local_train(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        本地训练
        
        Args:
            global_model_state: 全局模型状态字典
            
        Returns:
            model_update: 模型参数更新
        """
        # 加载全局模型参数
        self.model.load_state_dict(global_model_state)
        initial_state = copy.deepcopy(global_model_state)
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            
            for batch_idx, batch in enumerate(self.local_data):
                loss = self._train_batch(batch)
                batch_losses.append(loss)
                
                if batch_idx % 10 == 0:
                    self.logger.debug(
                        f'Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}'
                    )
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            self.logger.info(
                f'Client {self.client_id}, Epoch {epoch}, Average Loss: {epoch_loss:.4f}'
            )
        
        # 计算模型更新
        current_state = self.model.state_dict()
        model_update = {}
        for key in current_state:
            model_update[key] = current_state[key] - initial_state[key]
        
        # 记录训练统计
        self.training_stats['total_loss'].extend(epoch_losses)
        
        return model_update
    
    def _get_llm_data_for_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用大模型获取真实的logits数据
        
        Args:
            batch: 包含context_tokens的批次数据
            
        Returns:
            (context_embedding, top_k_logits, top_k_indices)
        """
        context_tokens = batch['context_tokens']  # [batch_size, seq_len]
        batch_size = context_tokens.size(0)
        top_k = self.config.get('top_k', 50)
        
        context_embeddings = []
        top_k_logits_list = []
        top_k_indices_list = []
        
        for i in range(batch_size):
            # 获取单个样本的context tokens
            sample_tokens = context_tokens[i].tolist()
            # 移除padding tokens (假设0是padding)
            sample_tokens = [t for t in sample_tokens if t != 0]
            
            # 获取模型期望的context_dim
            expected_context_dim = self.model.context_dim
        
            if self.llm_interface is not None:
                try:
                    # 使用真实大模型获取logits
                    full_logits = self.llm_interface.get_next_token_logits(sample_tokens)
                    top_k_logits, top_k_indices = torch.topk(full_logits, top_k)
                    
                    # 使用LLM接口获取有意义的上下文嵌入
                    context_embedding = self.llm_interface.get_context_embedding(sample_tokens)
                    
                except Exception as e:
                    self.logger.warning(f"使用LLM接口获取数据时出错: {e}，将回退到模拟数据")
                    # 回退到模拟数据
                    full_logits = torch.randn(self.config.get('vocab_size', 50000))
                    top_k_logits, top_k_indices = torch.topk(full_logits, top_k)
                    context_embedding = torch.randn(expected_context_dim)
            else:
                # 使用模拟数据
                full_logits = torch.randn(self.config.get('vocab_size', 50000))
                top_k_logits, top_k_indices = torch.topk(full_logits, top_k)
                context_embedding = torch.randn(expected_context_dim)
            
            context_embeddings.append(context_embedding)
            top_k_logits_list.append(top_k_logits)
            top_k_indices_list.append(top_k_indices)
        
        # 堆叠为批次张量
        context_embedding_batch = torch.stack(context_embeddings)
        top_k_logits_batch = torch.stack(top_k_logits_list)
        top_k_indices_batch = torch.stack(top_k_indices_list)
        
        return context_embedding_batch, top_k_logits_batch, top_k_indices_batch
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        训练单个批次
        
        Args:
            batch: 包含context_tokens等的批次数据
            
        Returns:
            loss: 批次损失
        """
        self.optimizer.zero_grad()
        
        # 获取模型设备
        device = next(self.model.parameters()).device
        
        # 获取大模型数据
        if 'context_embedding' in batch and 'top_k_logits' in batch and 'top_k_indices' in batch:
            # 如果批次中已经包含了预处理的数据，直接使用并移动到正确设备
            context_embedding = batch['context_embedding'].to(device)
            top_k_logits = batch['top_k_logits'].to(device)
            top_k_indices = batch['top_k_indices'].to(device)
        else:
            # 否则使用大模型获取数据
            context_embedding, top_k_logits, top_k_indices = self._get_llm_data_for_batch(batch)
            # 确保移动到正确设备
            context_embedding = context_embedding.to(device)
            top_k_logits = top_k_logits.to(device)
            top_k_indices = top_k_indices.to(device)
        
        context_tokens = batch['context_tokens'].to(device)  # [batch_size, seq_len]
        batch_size = context_embedding.size(0)
        
        # 生成目标水印比特并移动到正确设备
        target_bits = self.bit_generator.generate_bits(batch_size)
        target_bit_tensor = torch.tensor(target_bits, dtype=torch.long, device=device)
        
        # 生成绿名单mask并确保在正确设备上
        green_mask = self.greenlist_generator.get_greenlist_mask(context_tokens, top_k_indices)
        # 双重保险：确保green_mask在正确设备上
        green_mask = green_mask.to(device)
        
        # 前向传播
        selection_probs, selected_indices = self.model(
            context_embedding, top_k_logits, target_bit_tensor, top_k_indices
        )
        
        # 计算各项损失
        watermark_loss = self.model.get_watermark_loss(selection_probs, target_bit_tensor, green_mask)
        semantic_loss = self.model.get_semantic_loss(selection_probs, top_k_logits)
        fluency_loss = self.model.get_fluency_loss(selection_probs, top_k_logits)
        
        # 总损失
        total_loss = (
            self.lambda_watermark * watermark_loss +
            self.lambda_semantic * semantic_loss +
            self.lambda_fluency * fluency_loss
        )
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        # 记录损失
        self.training_stats['watermark_loss'].append(watermark_loss.item())
        self.training_stats['semantic_loss'].append(semantic_loss.item())
        self.training_stats['fluency_loss'].append(fluency_loss.item())
        
        return total_loss.item()
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """获取当前模型状态"""
        return self.model.state_dict()
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return self.training_stats


class FederatedAggregator(ABC):
    """联邦聚合器抽象基类"""
    
    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """聚合客户端更新"""
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