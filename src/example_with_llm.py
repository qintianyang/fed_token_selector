#!/usr/bin/env python3
"""
使用真实大模型的联邦学习Token选择器示例

这个脚本展示了如何：
1. 配置和使用不同类型的大模型接口
2. 在联邦学习中集成真实的大模型调用
3. 生成带水印的文本
"""

import torch
import logging
import yaml
from pathlib import Path

from llm_interface import LLMInterfaceFactory
from token_selector import TokenSelectorController
from federated_framework import FederatedClient, FederatedServer, FedAvgAggregator
from train_federated import FederatedTrainer, WatermarkDataset
from demo_complete import WatermarkTextGenerator
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_interfaces():
    """测试不同的大模型接口"""
    logger.info("=== 测试大模型接口 ===")
    
    # 测试HuggingFace接口
    try:
        logger.info("测试HuggingFace接口...")
        hf_config = {
            'model_name': 'meta-llama/Llama-3.2-1B',
            'config': {'device': 'cuda'}
        }
        hf_interface = LLMInterfaceFactory.create_interface('huggingface', **hf_config)
        
        # 测试获取logits
        test_tokens = [1, 2, 3, 4, 5]
        logits = hf_interface.get_next_token_logits(test_tokens)
        logger.info(f"HuggingFace logits形状: {logits.shape}")
        
        # 测试文本生成
        response = hf_interface.generate_text("Hello, world!", max_length=10)
        logger.info(f"HuggingFace生成文本: {response.text}")
        
    except Exception as e:
        logger.error(f"HuggingFace接口测试失败: {e}")
    
    # 测试OpenAI接口（需要API密钥）
    try:
        logger.info("测试OpenAI接口...")
        openai_config = {
            'model_name': 'gpt-3.5-turbo',
            'config': {
                'api_key': '',  # 需要设置真实的API密钥
                'base_url': 'https://api.openai.com/v1'
            }
        }
        
        if openai_config['config']['api_key']:
            openai_interface = LLMInterfaceFactory.create_interface('openai', **openai_config)
            
            test_tokens = [1, 2, 3, 4, 5]
            logits = openai_interface.get_next_token_logits(test_tokens)
            logger.info(f"OpenAI logits形状: {logits.shape}")
        else:
            logger.info("跳过OpenAI测试（未设置API密钥）")
            
    except Exception as e:
        logger.error(f"OpenAI接口测试失败: {e}")

def run_federated_training_with_llm():
    """使用YAML配置文件的联邦训练示例"""
    # 从YAML加载配置
    config = load_config_from_yaml()
    
    if config is None:
        # 如果YAML加载失败，使用硬编码配置作为备选
        config = {
            'vocab_size': 50257,
            'hidden_dim': 256,
            'top_k': 50,
            'gamma': 0.25,
            'learning_rate': 1e-4,
            'batch_size': 4,  # 较小的批次大小以减少计算量
            'samples_per_client': 50,  # 较少的样本数以加快演示
            'num_rounds': 5,  # 较少的轮数
            'lambda_watermark': 1.0,
            'lambda_semantic': 0.5,
            'lambda_fluency': 0.3,
            
            # 大模型配置
            'llm_config': {
                'enabled': True,
                'type': 'huggingface',
                'huggingface': {  # ✅ 正确：按接口类型分组配置
                    'model_name': 'meta-llama/Llama-3.2-1B',
                    'device': 'cuda',  # ✅ 正确：直接使用device
                    'cache_dir': '/home/qty/code/llama'  # 可选：模型缓存目录
                }
            }
        }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # try:
    # 创建联邦训练器
    trainer = FederatedTrainer(
        config=config,
        num_clients=2,  # 较少的客户端数量
        device=device,
        llm_config=config['llm_config']
    )
    
    # 运行几轮训练
    for round_num in range(1, config['num_rounds'] + 1):
        logger.info(f"开始第 {round_num} 轮训练")
        metrics = trainer.train_round(round_num)
        logger.info(f"第 {round_num} 轮完成，平均损失: {metrics['loss']:.4f}")
    
    # 保存训练好的模型
    model_path = 'trained_model_with_llm.pth'
    trainer.save_model(model_path)
    logger.info(f"模型已保存到: {model_path}")
        
    return trainer.global_model
        
    # except Exception as e:
    #     logger.error(f"联邦学习训练失败: {e}")
    #     return None

def test_watermark_generation_with_llm(model):
    """测试使用真实大模型的水印文本生成"""
    logger.info("=== 测试水印文本生成 ===")
    
    if model is None:
        logger.error("模型为空，跳过文本生成测试")
        return
    
    try:
        # 创建大模型接口
        llm_config = {
            'type': 'huggingface',
            'model_name': 'gpt2',
            'config': {'device': 'cpu'}
        }
        
        # 创建文本生成器
        generator = WatermarkTextGenerator(
            token_selector=model,
            vocab_size=50257,
            device='cpu',
            llm_config=llm_config
        )
        
        # 生成带水印的文本
        prompt_tokens = [15496, 11, 995]  # "Hello, world" 的近似token
        watermark_message = "SECRET"
        
        logger.info(f"生成带水印文本，水印消息: {watermark_message}")
        generated_tokens, embedded_bits = generator.generate_text_with_watermark(
            prompt_tokens=prompt_tokens,
            max_length=20,
            watermark_message=watermark_message
        )
        
        logger.info(f"生成的token序列长度: {len(generated_tokens)}")
        logger.info(f"嵌入的比特序列: {embedded_bits[:10]}...")  # 只显示前10个比特
        
    except Exception as e:
        logger.error(f"水印文本生成测试失败: {e}")

def load_config_from_yaml():
    """从YAML配置文件加载配置"""
    config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取相关配置
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        watermark_config = config.get('watermark', {})
        federated_config = config.get('federated', {})
        llm_config = config.get('llm_config', {})
        
        # 合并配置
        merged_config = {
            # 模型配置
            'vocab_size': model_config.get('vocab_size', 50000),
            'context_dim': model_config.get('context_dim', 768),
            'hidden_dim': model_config.get('hidden_dim', 256),
            'top_k': model_config.get('top_k', 50),
            'num_layers': model_config.get('num_transformer_layers', 3),
            'dropout': model_config.get('dropout', 0.1),
            
            # 训练配置
            'learning_rate': training_config.get('learning_rate', 1e-4),
            'batch_size': training_config.get('batch_size', 32),
            'local_epochs': training_config.get('local_epochs', 5),
            'samples_per_client': training_config.get('samples_per_client', 1000),
            
            # 水印配置
            'gamma': watermark_config.get('gamma', 0.25),
            
            # 联邦学习配置
            'num_rounds': federated_config.get('num_rounds', 100),
            'num_clients': federated_config.get('num_clients', 10),
            'client_fraction': federated_config.get('client_fraction', 1.0),
            
            # 大模型配置
            'llm_config': llm_config
        }
        
        logger.info("成功加载YAML配置文件")
        return merged_config
        
    except Exception as e:
        logger.warning(f"加载YAML配置失败: {e}，使用默认配置")
        return None

def main():
    """主函数"""
    logger.info("开始大模型集成示例")
    
    # 1. 测试大模型接口
    # test_llm_interfaces()
    
    # 2. 尝试从配置文件加载配置
    config = load_config_from_yaml()
    if config is None:
        logger.info("使用默认配置")
        # 使用简化的默认配置进行快速测试
        config = {
            'vocab_size': 1000,  # 较小的词汇表用于快速测试
            'hidden_dim': 64,
            'top_k': 20,
            'gamma': 0.25,
            'learning_rate': 1e-3,
            'batch_size': 2,
            'samples_per_client': 10,
            'num_rounds': 2,
            'lambda_watermark': 1.0,
            'lambda_semantic': 0.5,
            'lambda_fluency': 0.3,
            'llm_config': {
                'enabled': False  # 在快速测试中禁用真实大模型
            }
        }
    
    # 3. 运行联邦学习训练
    trained_model = run_federated_training_with_llm()
    
    # 4. 测试水印文本生成
    if trained_model is not None:
        test_watermark_generation_with_llm(trained_model)
    
    logger.info("示例完成")

if __name__ == '__main__':
    main()