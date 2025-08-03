import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import os

from token_selector import TokenSelectorController, WatermarkBitGenerator, GreenListGenerator
from federated_framework import FederatedClient, FederatedServer, FedAvgAggregator
from llm_interface import BaseLLMInterface, LLMInterfaceFactory

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkDataset(Dataset):
    """
    水印训练数据集
    模拟大模型的输出logits和对应的上下文
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 vocab_size: int = 50000,
                 max_seq_len: int = 128,
                 top_k: int = 50):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        
        # 生成模拟数据
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[Dict]:
        """生成合成训练数据"""
        data = []
        
        for i in range(self.num_samples):
            # 模拟上下文序列
            seq_len = np.random.randint(10, self.max_seq_len)
            context_tokens = torch.randint(0, self.vocab_size, (seq_len,))
            
            # 模拟大模型输出的logits (未归一化)
            logits = torch.randn(self.vocab_size) * 2.0
            
            # 获取top-k token和对应的logits
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
            
            # 随机生成目标水印比特
            target_bit = torch.randint(0, 2, (1,)).float()
            
            data.append({
                'context_tokens': context_tokens,
                'top_k_logits': top_k_logits,
                'top_k_indices': top_k_indices,
                'target_bit': target_bit
            })
            
        return data
    
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
        'top_k_logits': torch.stack([item['top_k_logits'] for item in batch]),
        'top_k_indices': torch.stack([item['top_k_indices'] for item in batch]),
        'target_bit': torch.stack([item['target_bit'] for item in batch])
    }

class FederatedTrainer:
    """
    联邦学习训练器
    """
    
    def __init__(self, 
                 config: Dict,
                 num_clients: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 llm_config: Optional[Dict] = None):
        self.config = config
        self.num_clients = num_clients
        self.device = device
        self.llm_config = llm_config or config.get('llm_config', {})
        
        # 初始化全局模型
        self.global_model = TokenSelectorController(
            vocab_size=config['vocab_size'],
            context_dim=config.get('context_dim', 768),
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            top_k=config['top_k']
        ).to(device)
        
        # 初始化联邦学习组件
        self.aggregator = FedAvgAggregator()
        self.server = FederatedServer(
            global_model=self.global_model,
            aggregator=self.aggregator,
            config=config
        )
        
        # 初始化客户端
        self.clients = []
        for i in range(num_clients):
            # 每个客户端有自己的数据集
            dataset = WatermarkDataset(
                num_samples=config['samples_per_client'],
                vocab_size=config['vocab_size'],
                top_k=config['top_k']
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=config['batch_size'],
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # 为每个客户端创建大模型接口（可选）
            client_llm_interface = None
            if self.llm_config.get('enabled', False):
                try:
                    interface_type = self.llm_config.get('type', 'huggingface')
                    
                    # 根据接口类型提取对应配置
                    if interface_type == 'huggingface':
                        hf_config = self.llm_config.get('huggingface', {})
                        client_llm_interface = LLMInterfaceFactory.create_interface(
                            'huggingface',
                            model_name=hf_config.get('model_name', 'gpt2'),
                            config={
                                'device': hf_config.get('device', 'auto'),
                                'cache_dir': hf_config.get('cache_dir', './models')
                            }
                        )
                    elif interface_type == 'openai':
                        openai_config = self.llm_config.get('openai', {})
                        client_llm_interface = LLMInterfaceFactory.create_interface(
                            'openai',
                            model_name=openai_config.get('model_name', 'gpt-3.5-turbo'),
                            config={
                                'api_key': openai_config.get('api_key', ''),
                                'base_url': openai_config.get('base_url', 'https://api.openai.com/v1'),
                                'num_samples': openai_config.get('num_samples', 50)
                            }
                        )
                    elif interface_type == 'local':
                        local_config = self.llm_config.get('local', {})
                        client_llm_interface = LLMInterfaceFactory.create_interface(
                            'local',
                            model_path=local_config.get('model_path', './models/local_model.pth'),
                            config={
                                'device': local_config.get('device', 'auto')
                            }
                        )
                    
                    logger.info(f"客户端 {i} 创建大模型接口成功")
                except Exception as e:
                    logger.warning(f"客户端 {i} 大模型接口创建失败: {e}")
            
            client = FederatedClient(
                client_id=i,
                model=TokenSelectorController(
                    vocab_size=config['vocab_size'],
                    context_dim=config.get('context_dim', 768),  # 添加context_dim参数
                    hidden_dim=config['hidden_dim'],
                    num_layers=config.get('num_layers', 3),      # 添加num_layers参数
                    dropout=config.get('dropout', 0.1),          # 添加dropout参数
                    top_k=config['top_k']
                ).to(device),
                local_data=dataloader,
                config=config,
                llm_interface=client_llm_interface
            )
            
            self.clients.append(client)
        
        # 绿名单生成器
        self.green_generator = GreenListGenerator(
            vocab_size=config['vocab_size'],
            gamma=config['gamma']
        )
        
        logger.info(f"初始化联邦训练器: {num_clients}个客户端, 设备: {device}")
        if self.llm_config.get('enabled', False):
            logger.info(f"大模型接口: {self.llm_config.get('type', 'huggingface')}")
    
    def train_round(self, round_num: int) -> Dict:
        """执行一轮联邦训练"""
        logger.info(f"开始第 {round_num} 轮联邦训练")
        
        # 1. 获取全局模型状态
        global_state = self.server.global_model.state_dict()
        
        # 2. 客户端本地训练
        client_updates = []
        client_weights = []
        client_metrics = []
        
        for client in self.clients:
            # 本地训练
            model_update = client.local_train(global_state)
            client_updates.append(model_update)
            
            # 客户端权重（基于数据量）
            client_weight = len(client.local_data.dataset)
            client_weights.append(client_weight)
            
            # 获取训练统计
            stats = client.get_training_stats()
            client_metrics.append({
                'loss': np.mean(stats['total_loss'][-client.local_epochs:]) if stats['total_loss'] else 0.0,
                'watermark_loss': np.mean(stats['watermark_loss'][-client.local_epochs:]) if stats['watermark_loss'] else 0.0,
                'semantic_loss': np.mean(stats['semantic_loss'][-client.local_epochs:]) if stats['semantic_loss'] else 0.0,
                'fluency_loss': np.mean(stats['fluency_loss'][-client.local_epochs:]) if stats['fluency_loss'] else 0.0
            })
        
        # 3. 聚合模型更新
        aggregated_update = self.server.aggregator.aggregate(client_updates, client_weights)
        
        # 4. 更新全局模型
        self.server._update_global_model(aggregated_update)
        
        # 5. 计算平均指标
        avg_metrics = self._average_metrics(client_metrics)
        
        logger.info(f"第 {round_num} 轮完成, 平均损失: {avg_metrics['loss']:.4f}")
        
        return avg_metrics
    
    def _train_client(self, client: FederatedClient) -> Tuple[Dict, Dict]:
        """训练单个客户端"""
        client.model.train()
        total_loss = 0.0
        total_watermark_loss = 0.0
        total_semantic_loss = 0.0
        total_fluency_loss = 0.0
        num_batches = 0
        
        for batch in client.train_loader:
            # 移动数据到设备
            context_tokens = batch['context_tokens'].to(self.device)
            top_k_logits = batch['top_k_logits'].to(self.device)
            top_k_indices = batch['top_k_indices'].to(self.device)
            target_bit = batch['target_bit'].to(self.device)
            
            # 生成绿名单mask
            green_mask = self.green_generator.get_greenlist_mask(
                context_tokens, top_k_indices
            ).to(self.device)
            
            # 前向传播
            selection_probs, selected_indices = client.model(
                context_tokens, top_k_logits, top_k_indices, target_bit
            )
            
            # 计算损失
            watermark_loss = client.model.get_watermark_loss(
                selection_probs, target_bit.squeeze(), green_mask
            )
            
            semantic_loss = client.model.get_semantic_loss(
                selection_probs, top_k_logits
            )
            
            fluency_loss = client.model.get_fluency_loss(
                selection_probs
            )
            
            # 总损失
            total_loss_batch = (
                self.config['lambda_watermark'] * watermark_loss +
                self.config['lambda_semantic'] * semantic_loss +
                self.config['lambda_fluency'] * fluency_loss
            )
            
            # 反向传播
            client.optimizer.zero_grad()
            total_loss_batch.backward()
            client.optimizer.step()
            
            # 累计损失
            total_loss += total_loss_batch.item()
            total_watermark_loss += watermark_loss.item()
            total_semantic_loss += semantic_loss.item()
            total_fluency_loss += fluency_loss.item()
            num_batches += 1
        
        # 获取模型更新
        update = client.get_model_update()
        
        # 计算平均损失
        metrics = {
            'loss': total_loss / num_batches,
            'watermark_loss': total_watermark_loss / num_batches,
            'semantic_loss': total_semantic_loss / num_batches,
            'fluency_loss': total_fluency_loss / num_batches
        }
        
        return update, metrics
    
    def _average_metrics(self, client_metrics: List[Dict]) -> Dict:
        """计算客户端指标的平均值"""
        avg_metrics = {}
        
        for key in client_metrics[0].keys():
            avg_metrics[key] = np.mean([metrics[key] for metrics in client_metrics])
        
        return avg_metrics
    
    def evaluate(self) -> Dict:
        """评估全局模型"""
        self.global_model.eval()
        
        # 创建测试数据集
        test_dataset = WatermarkDataset(
            num_samples=200,
            vocab_size=self.config['vocab_size'],
            top_k=self.config['top_k']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        total_loss = 0.0
        total_watermark_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                context_tokens = batch['context_tokens'].to(self.device)
                top_k_logits = batch['top_k_logits'].to(self.device)
                top_k_indices = batch['top_k_indices'].to(self.device)
                target_bit = batch['target_bit'].to(self.device)
                
                # 生成绿名单mask
                green_mask = self.green_generator.get_greenlist_mask(
                    context_tokens, top_k_indices
                ).to(self.device)
                
                # 前向传播
                selection_probs, selected_indices = self.global_model(
                    context_tokens, top_k_logits, top_k_indices, target_bit
                )
                
                # 计算损失
                watermark_loss = self.global_model.get_watermark_loss(
                    selection_probs, target_bit.squeeze(), green_mask
                )
                
                semantic_loss = self.global_model.get_semantic_loss(
                    selection_probs, top_k_logits
                )
                
                fluency_loss = self.global_model.get_fluency_loss(
                    selection_probs
                )
                
                total_loss_batch = (
                    self.config['lambda_watermark'] * watermark_loss +
                    self.config['lambda_semantic'] * semantic_loss +
                    self.config['lambda_fluency'] * fluency_loss
                )
                
                # 计算水印准确率
                green_probs = (selection_probs * green_mask).sum(dim=-1)
                watermark_pred = (green_probs > 0.5).float()
                watermark_accuracy = (watermark_pred == target_bit.squeeze()).float().mean()
                
                total_loss += total_loss_batch.item()
                total_watermark_accuracy += watermark_accuracy.item()
                num_batches += 1
        
        return {
            'test_loss': total_loss / num_batches,
            'watermark_accuracy': total_watermark_accuracy / num_batches
        }
    
    def save_model(self, path: str):
        """保存全局模型"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")

def main():
    """主训练函数"""
    # 训练配置
    config = {
        'vocab_size': 50000,
        'hidden_dim': 256,
        'top_k': 50,
        'gamma': 0.25,  # 绿名单比例
        'learning_rate': 1e-4,
        'batch_size': 16,
        'samples_per_client': 500,
        'num_rounds': 50,
        'lambda_watermark': 1.0,
        'lambda_semantic': 0.5,
        'lambda_fluency': 0.3
    }
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 初始化训练器
    trainer = FederatedTrainer(config, num_clients=5)
    
    # 训练循环
    training_history = []
    
    for round_num in range(1, config['num_rounds'] + 1):
        # 训练一轮
        metrics = trainer.train_round(round_num)
        training_history.append(metrics)
        
        # 每10轮评估一次
        if round_num % 10 == 0:
            eval_metrics = trainer.evaluate()
            logger.info(f"评估结果 - 测试损失: {eval_metrics['test_loss']:.4f}, "
                       f"水印准确率: {eval_metrics['watermark_accuracy']:.4f}")
            
            # 保存模型
            trainer.save_model(f'outputs/federated_model_round_{round_num}.pth')
    
    # 保存训练历史
    with open('outputs/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 最终评估
    final_metrics = trainer.evaluate()
    logger.info(f"最终评估结果: {final_metrics}")
    
    # 保存最终模型
    trainer.save_model('outputs/final_federated_model.pth')
    
    logger.info("联邦训练完成!")

if __name__ == '__main__':
    main()