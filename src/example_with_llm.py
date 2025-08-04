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
import os  # 添加这行
from pathlib import Path
from datetime import datetime  # 添加这行

from llm_interface import LLMInterfaceFactory
from token_selector import TokenSelectorController
from federated_framework import FederatedClient, FederatedServer, FedAvgAggregator
from train_federated import FederatedTrainer, WatermarkDataset
from demo_complete import WatermarkTextGenerator
from torch.utils.data import DataLoader
from config_manager import ConfigManager # 导入 ConfigManager

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

# 在文件开头添加 os 导入
import os

# 在 run_federated_training_with_llm 函数中，保存模型之前添加：
def run_federated_training_with_llm():
    """使用YAML配置文件的联邦训练示例"""
    # 使用 ConfigManager 加载和管理配置
    config_manager = ConfigManager()
    
    # 获取扁平化的训练配置
    training_config = config_manager.create_training_config_dict()
    
    # 从原始配置中获取 llm_config 和 data_paths
    llm_config = config_manager.get('llm_config', {})
    data_paths = config_manager.get('data_paths') # 现在可以正确获取
    num_clients = config_manager.get('federated.num_clients', 2)

    # 将 num_clients 和 num_rounds 添加到扁平化配置中，因为 FederatedTrainer 可能需要它们
    training_config['num_clients'] = num_clients
    training_config['num_rounds'] = config_manager.get('federated.num_rounds', 1)
    
    device = config_manager.get('experiment.device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建联邦训练器
    trainer = FederatedTrainer(
        config=training_config, # 使用扁平化的配置
        num_clients=num_clients,
        device=device,
        llm_config=llm_config,
        data_paths=data_paths  # 传入数据路径
    )
    
    # 运行几轮训练
    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 创建训练结果记录文件
    results_file = 'outputs/training_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("联邦学习训练结果记录\n")
        f.write("=" * 50 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总轮数: {training_config['num_rounds']}\n")
        f.write(f"客户端数量: {training_config['num_clients']}\n")
        f.write("\n")
    
    # 运行训练循环
    for round_num in range(1, training_config['num_rounds'] + 1):
        logger.info(f"开始第 {round_num} 轮训练")
        metrics = trainer.train_round(round_num)
        
        # 打印详细的训练指标
        logger.info(f"第 {round_num} 轮训练完成:")
        logger.info(f"  总损失: {metrics['loss']:.4f}")
        logger.info(f"  水印损失: {metrics['watermark_loss']:.4f}")
        logger.info(f"  语义损失: {metrics['semantic_loss']:.4f}")
        logger.info(f"  流畅性损失: {metrics['fluency_loss']:.4f}")
        
        # 保存每轮的模型
        model_path = f'outputs/model_round_{round_num}.pth'
        # 确保目录存在（双重保险）
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
        logger.info(f"第 {round_num} 轮模型已保存到: {model_path}")
        
        # 将结果追加到txt文件
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"第 {round_num} 轮训练结果:\n")
            f.write(f"  总损失: {metrics['loss']:.6f}\n")
            f.write(f"  水印损失: {metrics['watermark_loss']:.6f}\n")
            f.write(f"  语义损失: {metrics['semantic_loss']:.6f}\n")
            f.write(f"  流畅性损失: {metrics['fluency_loss']:.6f}\n")
            f.write(f"  模型保存路径: {model_path}\n")
            f.write(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")
    
    # 记录训练完成信息
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
    
    logger.info(f"训练结果已保存到: {results_file}")
    logger.info(f"第 {round_num} 轮完成，平均损失: {metrics['loss']:.4f}")
    
    # 保存训练好的模型
    model_path = 'trained_model_with_llm.pth'
    trainer.save_model(model_path)
    logger.info(f"模型已保存到: {model_path}")
        
    return trainer.global_model
        

def test_watermark_generation_with_llm(model):
    """测试使用真实大模型的水印文本生成"""
    logger.info("=== 测试水印文本生成 ===")
    
    if model is None:
        logger.error("模型为空，跳过文本生成测试")
        return
    
    try:
        # 自动检测可用设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"水印生成测试使用设备: {device}")

        # 创建文本生成器，统一使用检测到的设备
        generator = WatermarkTextGenerator(
            token_selector=model,
            vocab_size=50257, # GPT-2/Llama vocab size
            device=device,
            llm_config={
                'type': 'huggingface',
                'model_name': 'meta-llama/Llama-3.2-1B',
                # 'config'中的device将由WatermarkTextGenerator内部设置，此处可省略
            }
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
    
    # 3. 运行联邦学习训练
    trained_model = run_federated_training_with_llm()
    
    # 4. 测试水印文本生成
    if trained_model is not None:
        test_watermark_generation_with_llm(trained_model)
    
    logger.info("示例完成")

if __name__ == '__main__':
    main()