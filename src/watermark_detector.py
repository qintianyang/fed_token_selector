import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from scipy import stats
import logging

from token_selector import GreenListGenerator

logger = logging.getLogger(__name__)

class WatermarkDetector:
    """
    水印检测器 - 用于检测文本中的水印比特序列
    
    基于统计方法检测token序列中是否包含水印：
    1. 重构每个位置的绿名单
    2. 统计绿名单token的出现频率
    3. 使用z-score检验判断是否存在水印
    """
    
    def __init__(self, 
                 vocab_size: int,
                 gamma: float = 0.25,
                 z_threshold: float = 4.0,
                 hash_key: int = 15485863):
        """
        Args:
            vocab_size: 词汇表大小
            gamma: 绿名单比例
            z_threshold: z-score阈值，超过此值认为存在水印
            hash_key: 哈希密钥，需要与生成时一致
        """
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.z_threshold = z_threshold
        self.hash_key = hash_key
        
        # 绿名单生成器
        self.green_generator = GreenListGenerator(
            vocab_size=vocab_size,
            gamma=gamma,
            hash_key=hash_key
        )
        
        logger.info(f"初始化水印检测器: gamma={gamma}, z_threshold={z_threshold}")
    
    def detect_watermark(self, 
                        token_sequence: List[int],
                        return_details: bool = False) -> Dict:
        """
        检测token序列中的水印
        
        Args:
            token_sequence: token序列
            return_details: 是否返回详细信息
            
        Returns:
            检测结果字典，包含：
            - is_watermarked: 是否包含水印
            - z_score: z-score值
            - p_value: p值
            - green_token_count: 绿名单token数量
            - total_tokens: 总token数量
            - green_fraction: 绿名单token比例
        """
        if len(token_sequence) < 2:
            raise ValueError("序列长度至少需要2个token")
        
        # 统计绿名单token
        green_token_count = 0
        green_token_mask = []
        
        # 从第二个token开始检测（第一个token没有前缀）
        for i in range(1, len(token_sequence)):
            # 获取前缀
            prefix_tokens = torch.tensor(token_sequence[:i])
            current_token = token_sequence[i]
            
            # 生成绿名单
            greenlist_ids = self._get_greenlist_ids(prefix_tokens)
            
            # 检查当前token是否在绿名单中
            if current_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)
        
        # 计算统计量
        num_tokens_scored = len(token_sequence) - 1  # 排除第一个token
        green_fraction = green_token_count / num_tokens_scored if num_tokens_scored > 0 else 0
        
        # 计算z-score
        z_score = self._calculate_z_score(green_token_count, num_tokens_scored, self.gamma)
        
        # 计算p值
        p_value = self._calculate_p_value(z_score)
        
        # 判断是否包含水印
        is_watermarked = z_score > self.z_threshold
        
        result = {
            'is_watermarked': is_watermarked,
            'z_score': z_score,
            'p_value': p_value,
            'green_token_count': green_token_count,
            'total_tokens': num_tokens_scored,
            'green_fraction': green_fraction,
            'expected_green_fraction': self.gamma
        }
        
        if return_details:
            result['green_token_mask'] = green_token_mask
            result['token_sequence'] = token_sequence
        
        return result
    
    def _get_greenlist_ids(self, prefix_tokens: torch.Tensor) -> List[int]:
        """
        基于前缀生成绿名单token ID
        
        Args:
            prefix_tokens: 前缀token序列
            
        Returns:
            绿名单token ID列表
        """
        if len(prefix_tokens) == 0:
            seed = self.hash_key
        else:
            seed = self.hash_key * prefix_tokens[-1].item()
        
        # 生成绿名单
        rng = torch.Generator()
        rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=rng)
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_ids = vocab_permutation[:greenlist_size].tolist()
        
        return greenlist_ids
    
    def _calculate_z_score(self, 
                          green_count: int, 
                          total_count: int, 
                          expected_ratio: float) -> float:
        """
        计算z-score
        
        在零假设下（无水印），绿名单token的出现应该服从二项分布B(n, gamma)
        使用正态近似计算z-score
        """
        if total_count == 0:
            return 0.0
        
        # 期望值和方差
        expected_count = total_count * expected_ratio
        variance = total_count * expected_ratio * (1 - expected_ratio)
        
        if variance == 0:
            return 0.0
        
        # z-score计算
        z_score = (green_count - expected_count) / math.sqrt(variance)
        
        return z_score
    
    def _calculate_p_value(self, z_score: float) -> float:
        """
        计算p值（单尾检验）
        """
        # 使用标准正态分布的累积分布函数
        p_value = 1 - stats.norm.cdf(z_score)
        return p_value
    
    def detect_bit_sequence(self, 
                           token_sequence: List[int],
                           bit_length: int = 32) -> Dict:
        """
        尝试从token序列中提取水印比特序列
        
        Args:
            token_sequence: token序列
            bit_length: 期望的比特序列长度
            
        Returns:
            提取结果，包含比特序列和置信度
        """
        if len(token_sequence) < bit_length + 1:
            raise ValueError(f"序列长度不足，需要至少{bit_length + 1}个token")
        
        extracted_bits = []
        confidence_scores = []
        
        # 从第二个token开始提取比特
        for i in range(1, min(len(token_sequence), bit_length + 1)):
            prefix_tokens = torch.tensor(token_sequence[:i])
            current_token = token_sequence[i]
            
            # 生成绿名单
            greenlist_ids = self._get_greenlist_ids(prefix_tokens)
            
            # 判断当前token是否在绿名单中
            if current_token in greenlist_ids:
                extracted_bits.append(1)
                confidence_scores.append(1.0)  # 在绿名单中，置信度高
            else:
                extracted_bits.append(0)
                confidence_scores.append(1.0)  # 不在绿名单中，置信度也高
        
        # 计算整体置信度
        overall_confidence = np.mean(confidence_scores)
        
        return {
            'extracted_bits': extracted_bits,
            'bit_string': ''.join(map(str, extracted_bits)),
            'confidence_scores': confidence_scores,
            'overall_confidence': overall_confidence,
            'bit_length': len(extracted_bits)
        }
    
    def batch_detect(self, 
                    token_sequences: List[List[int]],
                    return_details: bool = False) -> List[Dict]:
        """
        批量检测多个token序列
        
        Args:
            token_sequences: token序列列表
            return_details: 是否返回详细信息
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i, sequence in enumerate(token_sequences):
            try:
                result = self.detect_watermark(sequence, return_details)
                result['sequence_id'] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"检测序列{i}时出错: {e}")
                results.append({
                    'sequence_id': i,
                    'is_watermarked': False,
                    'error': str(e)
                })
        
        return results
    
    def get_detection_stats(self, results: List[Dict]) -> Dict:
        """
        计算检测统计信息
        
        Args:
            results: 检测结果列表
            
        Returns:
            统计信息字典
        """
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': '没有有效的检测结果'}
        
        watermarked_count = sum(1 for r in valid_results if r['is_watermarked'])
        total_count = len(valid_results)
        
        z_scores = [r['z_score'] for r in valid_results]
        p_values = [r['p_value'] for r in valid_results]
        green_fractions = [r['green_fraction'] for r in valid_results]
        
        stats_dict = {
            'total_sequences': total_count,
            'watermarked_sequences': watermarked_count,
            'watermark_rate': watermarked_count / total_count,
            'avg_z_score': np.mean(z_scores),
            'std_z_score': np.std(z_scores),
            'avg_p_value': np.mean(p_values),
            'avg_green_fraction': np.mean(green_fractions),
            'expected_green_fraction': self.gamma
        }
        
        return stats_dict

class WatermarkEvaluator:
    """
    水印评估器 - 用于评估水印系统的性能
    """
    
    def __init__(self, detector: WatermarkDetector):
        self.detector = detector
    
    def evaluate_detection_performance(self, 
                                     watermarked_sequences: List[List[int]],
                                     unwatermarked_sequences: List[List[int]]) -> Dict:
        """
        评估检测性能
        
        Args:
            watermarked_sequences: 带水印的序列
            unwatermarked_sequences: 不带水印的序列
            
        Returns:
            性能指标字典
        """
        # 检测带水印序列
        watermarked_results = self.detector.batch_detect(watermarked_sequences)
        
        # 检测不带水印序列
        unwatermarked_results = self.detector.batch_detect(unwatermarked_sequences)
        
        # 计算性能指标
        tp = sum(1 for r in watermarked_results if r.get('is_watermarked', False))
        fn = len(watermarked_results) - tp
        
        fp = sum(1 for r in unwatermarked_results if r.get('is_watermarked', False))
        tn = len(unwatermarked_results) - fp
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'watermarked_stats': self.detector.get_detection_stats(watermarked_results),
            'unwatermarked_stats': self.detector.get_detection_stats(unwatermarked_results)
        }
    
    def analyze_z_score_distribution(self, 
                                   sequences: List[List[int]],
                                   labels: List[bool]) -> Dict:
        """
        分析z-score分布
        
        Args:
            sequences: token序列列表
            labels: 真实标签（True表示带水印）
            
        Returns:
            分布分析结果
        """
        results = self.detector.batch_detect(sequences)
        
        watermarked_z_scores = []
        unwatermarked_z_scores = []
        
        for result, label in zip(results, labels):
            if 'z_score' in result:
                if label:
                    watermarked_z_scores.append(result['z_score'])
                else:
                    unwatermarked_z_scores.append(result['z_score'])
        
        return {
            'watermarked_z_scores': watermarked_z_scores,
            'unwatermarked_z_scores': unwatermarked_z_scores,
            'watermarked_mean': np.mean(watermarked_z_scores) if watermarked_z_scores else 0,
            'watermarked_std': np.std(watermarked_z_scores) if watermarked_z_scores else 0,
            'unwatermarked_mean': np.mean(unwatermarked_z_scores) if unwatermarked_z_scores else 0,
            'unwatermarked_std': np.std(unwatermarked_z_scores) if unwatermarked_z_scores else 0
        }

def demo_detection():
    """
    水印检测演示
    """
    # 初始化检测器
    detector = WatermarkDetector(
        vocab_size=50000,
        gamma=0.25,
        z_threshold=4.0
    )
    
    # 模拟一个带水印的序列
    # 这里简单模拟，实际应该是由训练好的模型生成
    watermarked_sequence = [1234, 5678, 9012, 3456, 7890, 1111, 2222, 3333, 4444, 5555]
    
    # 检测水印
    result = detector.detect_watermark(watermarked_sequence, return_details=True)
    
    print("水印检测结果:")
    print(f"是否包含水印: {result['is_watermarked']}")
    print(f"Z-score: {result['z_score']:.4f}")
    print(f"P值: {result['p_value']:.6f}")
    print(f"绿名单token数量: {result['green_token_count']}/{result['total_tokens']}")
    print(f"绿名单比例: {result['green_fraction']:.4f} (期望: {result['expected_green_fraction']:.4f})")
    
    # 尝试提取比特序列
    bit_result = detector.detect_bit_sequence(watermarked_sequence, bit_length=8)
    print(f"\n提取的比特序列: {bit_result['bit_string']}")
    print(f"置信度: {bit_result['overall_confidence']:.4f}")

if __name__ == '__main__':
    demo_detection()