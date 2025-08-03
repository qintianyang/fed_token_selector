#!/usr/bin/env python3
"""
联邦学习Token选择器水印系统完整演示

这个脚本展示了：
1. 联邦学习训练token选择器
2. 使用训练好的模型生成带水印的文本
3. 水印检测和验证
4. 性能评估
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import logging
import json
import os
from tqdm import tqdm

from token_selector import TokenSelectorController, WatermarkBitGenerator, GreenListGenerator
from federated_framework import FederatedClient, FederatedServer, FedAvgAggregator
from watermark_detector import WatermarkDetector, WatermarkEvaluator
from train_federated import WatermarkDataset, FederatedTrainer, collate_fn

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkTextGenerator:
    """
    带水印的文本生成器
    使用训练好的token选择器生成带水印的文本
    """
    
    def __init__(self, 
                 token_selector: TokenSelectorController,
                 vocab_size: int,
                 device: str = 'cpu'):
        self.token_selector = token_selector
        self.vocab_size = vocab_size
        self.device = device
        
        # 水印比特生成器
        self.bit_generator = WatermarkBitGenerator()
        
        # 绿名单生成器
        self.green_generator = GreenListGenerator(vocab_size=vocab_size)
        
        self.token_selector.eval()
    
    def generate_text_with_watermark(self, 
                                   prompt_tokens: List[int],
                                   max_length: int = 50,
                                   watermark_message: str = "SECRET",
                                   top_k: int = 50) -> Tuple[List[int], List[int]]:
        """
        生成带水印的文本
        
        Args:
            prompt_tokens: 初始prompt的token序列
            max_length: 最大生成长度
            watermark_message: 要嵌入的水印消息
            top_k: top-k采样参数
            
        Returns:
            (生成的token序列, 嵌入的比特序列)
        """
        # 将水印消息转换为比特序列
        watermark_bits = self.bit_generator.message_to_bits(watermark_message)
        
        generated_tokens = prompt_tokens.copy()
        embedded_bits = []
        
        with torch.no_grad():
            for step in range(max_length):
                # 获取当前上下文
                context_tokens = torch.tensor([generated_tokens]).to(self.device)
                
                # 模拟大模型输出logits
                logits = self._simulate_llm_logits(context_tokens)
                
                # 获取top-k
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                
                # 获取当前要嵌入的比特
                current_bit_idx = step % len(watermark_bits)
                target_bit = torch.tensor([watermark_bits[current_bit_idx]]).float().to(self.device)
                
                # 使用token选择器选择token
                selection_probs, selected_indices = self.token_selector(
                    context_tokens, 
                    top_k_logits.unsqueeze(0), 
                    top_k_indices.unsqueeze(0), 
                    target_bit
                )
                
                # 选择token
                selected_token = selected_indices[0].item()
                generated_tokens.append(selected_token)
                embedded_bits.append(watermark_bits[current_bit_idx])
                
                # 简单的停止条件（可以根据需要修改）
                if selected_token == 0:  # 假设0是EOS token
                    break
        
        return generated_tokens, embedded_bits
    
    def _simulate_llm_logits(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        模拟大语言模型的logits输出
        在实际应用中，这里应该调用真实的LLM API
        """
        # 简单的模拟：基于上下文生成随机但有一定模式的logits
        batch_size, seq_len = context_tokens.shape
        
        # 使用上下文的最后一个token作为种子
        if seq_len > 0:
            seed = context_tokens[0, -1].item()
        else:
            seed = 42
        
        torch.manual_seed(seed)
        logits = torch.randn(self.vocab_size) * 2.0
        
        # 添加一些偏置，使某些token更可能被选择
        high_prob_tokens = torch.randint(0, self.vocab_size, (100,))
        logits[high_prob_tokens] += 1.0
        
        return logits
    
    def generate_batch_with_watermark(self, 
                                    prompts: List[List[int]],
                                    watermark_messages: List[str],
                                    max_length: int = 50) -> Tuple[List[List[int]], List[List[int]]]:
        """
        批量生成带水印的文本
        """
        generated_sequences = []
        embedded_bit_sequences = []
        
        for prompt, message in zip(prompts, watermark_messages):
            tokens, bits = self.generate_text_with_watermark(
                prompt, max_length, message
            )
            generated_sequences.append(tokens)
            embedded_bit_sequences.append(bits)
        
        return generated_sequences, embedded_bit_sequences

def run_complete_demo():
    """
    运行完整的演示
    """
    logger.info("开始联邦学习Token选择器水印系统演示")
    
    # 配置参数
    config = {
        'vocab_size': 10000,  # 为了演示，使用较小的词汇表
        'hidden_dim': 128,
        'top_k': 20,
        'gamma': 0.25,
        'learning_rate': 1e-3,
        'batch_size': 8,
        'samples_per_client': 100,
        'num_rounds': 20,  # 为了演示，使用较少的轮数
        'lambda_watermark': 1.0,
        'lambda_semantic': 0.5,
        'lambda_fluency': 0.3
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs('demo_outputs', exist_ok=True)
    
    # ==================== 第一阶段：联邦学习训练 ====================
    logger.info("第一阶段：开始联邦学习训练")
    
    trainer = FederatedTrainer(config, num_clients=3, device=device)
    
    training_history = []
    for round_num in range(1, config['num_rounds'] + 1):
        metrics = trainer.train_round(round_num)
        training_history.append(metrics)
        
        if round_num % 5 == 0:
            eval_metrics = trainer.evaluate()
            logger.info(f"轮次 {round_num} - 测试损失: {eval_metrics['test_loss']:.4f}, "
                       f"水印准确率: {eval_metrics['watermark_accuracy']:.4f}")
    
    # 保存训练好的模型
    trainer.save_model('demo_outputs/trained_model.pth')
    
    # ==================== 第二阶段：生成带水印文本 ====================
    logger.info("第二阶段：生成带水印文本")
    
    # 创建文本生成器
    generator = WatermarkTextGenerator(
        token_selector=trainer.global_model,
        vocab_size=config['vocab_size'],
        device=device
    )
    
    # 准备测试数据
    test_prompts = [
        [1, 2, 3, 4, 5],  # 模拟prompt
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500],
        [1000, 2000, 3000, 4000, 5000],
        [42, 123, 456, 789, 101]
    ]
    
    watermark_messages = [
        "HELLO",
        "WORLD", 
        "SECRET",
        "DEMO",
        "TEST"
    ]
    
    # 生成带水印的文本
    watermarked_sequences, embedded_bits = generator.generate_batch_with_watermark(
        test_prompts, watermark_messages, max_length=30
    )
    
    # 生成不带水印的文本（用于对比）
    unwatermarked_sequences = []
    for prompt in test_prompts:
        # 简单的随机生成（模拟无水印文本）
        sequence = prompt.copy()
        for _ in range(30):
            next_token = np.random.randint(0, config['vocab_size'])
            sequence.append(next_token)
        unwatermarked_sequences.append(sequence)
    
    logger.info(f"生成了 {len(watermarked_sequences)} 个带水印序列")
    logger.info(f"生成了 {len(unwatermarked_sequences)} 个无水印序列")
    
    # ==================== 第三阶段：水印检测 ====================
    logger.info("第三阶段：水印检测")
    
    # 创建水印检测器
    detector = WatermarkDetector(
        vocab_size=config['vocab_size'],
        gamma=config['gamma'],
        z_threshold=2.0  # 降低阈值以便演示
    )
    
    # 检测带水印文本
    watermarked_results = detector.batch_detect(watermarked_sequences, return_details=True)
    
    # 检测无水印文本
    unwatermarked_results = detector.batch_detect(unwatermarked_sequences, return_details=True)
    
    # ==================== 第四阶段：性能评估 ====================
    logger.info("第四阶段：性能评估")
    
    evaluator = WatermarkEvaluator(detector)
    
    # 评估检测性能
    performance = evaluator.evaluate_detection_performance(
        watermarked_sequences, unwatermarked_sequences
    )
    
    logger.info("检测性能评估结果:")
    logger.info(f"准确率: {performance['accuracy']:.4f}")
    logger.info(f"精确率: {performance['precision']:.4f}")
    logger.info(f"召回率: {performance['recall']:.4f}")
    logger.info(f"F1分数: {performance['f1_score']:.4f}")
    
    # ==================== 第五阶段：结果可视化 ====================
    logger.info("第五阶段：结果可视化")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    
    # 训练损失曲线
    plt.subplot(2, 3, 1)
    rounds = list(range(1, len(training_history) + 1))
    losses = [h['loss'] for h in training_history]
    plt.plot(rounds, losses, 'b-', linewidth=2)
    plt.title('训练损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid(True)
    
    # 各组件损失
    plt.subplot(2, 3, 2)
    watermark_losses = [h['watermark_loss'] for h in training_history]
    semantic_losses = [h['semantic_loss'] for h in training_history]
    fluency_losses = [h['fluency_loss'] for h in training_history]
    
    plt.plot(rounds, watermark_losses, 'r-', label='水印损失', linewidth=2)
    plt.plot(rounds, semantic_losses, 'g-', label='语义损失', linewidth=2)
    plt.plot(rounds, fluency_losses, 'b-', label='流畅性损失', linewidth=2)
    plt.title('各组件损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # Z-score分布
    plt.subplot(2, 3, 3)
    watermarked_z_scores = [r['z_score'] for r in watermarked_results if 'z_score' in r]
    unwatermarked_z_scores = [r['z_score'] for r in unwatermarked_results if 'z_score' in r]
    
    plt.hist(watermarked_z_scores, bins=10, alpha=0.7, label='带水印', color='red')
    plt.hist(unwatermarked_z_scores, bins=10, alpha=0.7, label='无水印', color='blue')
    plt.axvline(x=detector.z_threshold, color='black', linestyle='--', label='检测阈值')
    plt.title('Z-score分布')
    plt.xlabel('Z-score')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True)
    
    # 绿名单比例分布
    plt.subplot(2, 3, 4)
    watermarked_green_fractions = [r['green_fraction'] for r in watermarked_results if 'green_fraction' in r]
    unwatermarked_green_fractions = [r['green_fraction'] for r in unwatermarked_results if 'green_fraction' in r]
    
    plt.hist(watermarked_green_fractions, bins=10, alpha=0.7, label='带水印', color='red')
    plt.hist(unwatermarked_green_fractions, bins=10, alpha=0.7, label='无水印', color='blue')
    plt.axvline(x=config['gamma'], color='black', linestyle='--', label='期望比例')
    plt.title('绿名单Token比例分布')
    plt.xlabel('绿名单比例')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True)
    
    # 混淆矩阵
    plt.subplot(2, 3, 5)
    confusion_matrix = np.array([
        [performance['true_positives'], performance['false_negatives']],
        [performance['false_positives'], performance['true_negatives']]
    ])
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测无水印', '预测有水印'],
                yticklabels=['实际有水印', '实际无水印'])
    plt.title('混淆矩阵')
    
    # 性能指标
    plt.subplot(2, 3, 6)
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [performance['accuracy'], performance['precision'], 
              performance['recall'], performance['f1_score']]
    
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('检测性能指标')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('demo_outputs/demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== 保存详细结果 ====================
    logger.info("保存详细结果")
    
    # 保存检测结果
    results_summary = {
        'config': config,
        'training_history': training_history,
        'performance_metrics': performance,
        'watermarked_detection_stats': detector.get_detection_stats(watermarked_results),
        'unwatermarked_detection_stats': detector.get_detection_stats(unwatermarked_results)
    }
    
    with open('demo_outputs/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # 保存示例序列和检测结果
    examples = {
        'watermarked_sequences': watermarked_sequences[:3],  # 保存前3个示例
        'unwatermarked_sequences': unwatermarked_sequences[:3],
        'embedded_bits': embedded_bits[:3],
        'watermark_messages': watermark_messages[:3],
        'watermarked_detection_results': watermarked_results[:3],
        'unwatermarked_detection_results': unwatermarked_results[:3]
    }
    
    with open('demo_outputs/examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    # ==================== 比特序列提取演示 ====================
    logger.info("比特序列提取演示")
    
    for i, (seq, original_bits, message) in enumerate(zip(watermarked_sequences[:3], embedded_bits[:3], watermark_messages[:3])):
        logger.info(f"\n序列 {i+1} (原始消息: '{message}'):")
        
        # 提取比特序列
        bit_result = detector.detect_bit_sequence(seq, bit_length=len(original_bits))
        
        logger.info(f"原始比特: {''.join(map(str, original_bits))}")
        logger.info(f"提取比特: {bit_result['bit_string']}")
        logger.info(f"匹配度: {sum(a == b for a, b in zip(original_bits, bit_result['extracted_bits'])) / len(original_bits):.2%}")
        logger.info(f"置信度: {bit_result['overall_confidence']:.4f}")
    
    logger.info("\n演示完成！结果已保存到 demo_outputs/ 目录")
    logger.info("主要文件:")
    logger.info("- trained_model.pth: 训练好的模型")
    logger.info("- results_summary.json: 完整结果摘要")
    logger.info("- examples.json: 示例序列和检测结果")
    logger.info("- demo_results.png: 可视化结果")

def quick_test():
    """
    快速测试 - 用于验证代码是否正常工作
    """
    logger.info("运行快速测试...")
    
    # 简化的配置
    config = {
        'vocab_size': 1000,
        'hidden_dim': 64,
        'top_k': 10,
        'gamma': 0.25,
        'learning_rate': 1e-2,
        'batch_size': 4,
        'samples_per_client': 20,
        'num_rounds': 3,
        'lambda_watermark': 1.0,
        'lambda_semantic': 0.5,
        'lambda_fluency': 0.3
    }
    
    device = 'cpu'  # 强制使用CPU以确保兼容性
    
    # 快速训练
    trainer = FederatedTrainer(config, num_clients=2, device=device)
    
    for round_num in range(1, config['num_rounds'] + 1):
        metrics = trainer.train_round(round_num)
        logger.info(f"轮次 {round_num} - 损失: {metrics['loss']:.4f}")
    
    # 快速检测测试
    detector = WatermarkDetector(
        vocab_size=config['vocab_size'],
        gamma=config['gamma']
    )
    
    test_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = detector.detect_watermark(test_sequence)
    
    logger.info(f"检测结果: {result['is_watermarked']}, Z-score: {result['z_score']:.4f}")
    logger.info("快速测试完成！")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='联邦学习Token选择器水印系统演示')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='运行模式: full=完整演示, quick=快速测试')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_complete_demo()
    else:
        quick_test()