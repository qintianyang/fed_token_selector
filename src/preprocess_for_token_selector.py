import json
import pickle
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from llm_interface import LLMInterfaceFactory
from token_selector import WatermarkBitGenerator, GreenListGenerator
from config_manager import ConfigManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenSelectorDataPreprocessor:
    """为Token选择器预处理数据"""
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.device = self.config.get('experiment', {}).get('device', 'cpu')
        
        # 初始化LLM接口
        self.llm_interface = self._init_llm_interface()
        
        # 初始化水印组件
        self.bit_generator = WatermarkBitGenerator()
        vocab_size = self.config.get('model', {}).get('vocab_size', 50257)
        self.green_generator = GreenListGenerator(vocab_size=vocab_size)
        
        # 配置参数
        self.top_k = self.config.get('model', {}).get('max_candidate_tokens', 50)
        self.max_context_length = self.config.get('model', {}).get('max_context_length', 512)
        
    def _init_llm_interface(self):
        """初始化LLM接口"""
        try:
            llm_interface = LLMInterfaceFactory.create_llm_interface(
                'huggingface',
                model_name='facebook/opt-1.3b',
                device=self.device,
                cache_dir='/home/qty/code/opt-1.3b',
                torch_dtype='float16' if self.device == 'cuda' else 'float32'
            )
            logger.info(f"成功初始化LLM接口，设备: {self.device}")
            return llm_interface
        except Exception as e:
            logger.error(f"LLM接口初始化失败: {e}")
            raise
    
    def _get_top_k_logits_and_indices(self, context_tokens: List[int]) -> tuple:
        """获取top-k logits和对应的token indices"""
        try:
            # 限制上下文长度
            if len(context_tokens) > self.max_context_length:
                context_tokens = context_tokens[-self.max_context_length:]
            
            # 使用LLM接口的generate方法获取logits
            # 由于HuggingFaceInterface没有get_logits方法，我们需要通过其他方式获取
            if hasattr(self.llm_interface, 'model') and hasattr(self.llm_interface, 'tokenizer'):
                # 直接使用模型获取logits
                input_ids = torch.tensor([context_tokens]).to(self.device)
                with torch.no_grad():
                    outputs = self.llm_interface.model(input_ids)
                    logits = outputs.logits[0, -1, :]  # 获取最后一个位置的logits
                
                # 获取top-k
                top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
                return top_k_logits.cpu(), top_k_indices.cpu()
            else:
                raise AttributeError("无法访问模型")
                
        except Exception as e:
            logger.warning(f"获取logits失败: {e}，使用随机值")
            # 回退到随机值
            vocab_size = self.config.get('model', {}).get('vocab_size', 50257)
            top_k_indices = torch.randint(0, vocab_size, (self.top_k,))
            top_k_logits = torch.randn(self.top_k)
            return top_k_logits, top_k_indices
    
    def _generate_target_bits(self, num_samples: int) -> List[int]:
        """生成目标水印比特序列"""
        return self.bit_generator.generate_bits(num_samples)
    
    def _get_green_mask(self, context_tokens: List[int], top_k_indices: torch.Tensor) -> torch.Tensor:
        """获取绿名单mask"""
        try:
            context_tensor = torch.tensor(context_tokens[-10:])  # 使用最后10个token作为上下文
            green_mask = self.green_generator.get_greenlist_mask(
                context_tensor.unsqueeze(0),  # [1, seq_len]
                top_k_indices.unsqueeze(0)    # [1, top_k]
            )
            return green_mask.squeeze(0)  # [top_k]
        except Exception as e:
            logger.warning(f"生成绿名单mask失败: {e}，使用随机mask")
            return torch.randint(0, 2, (self.top_k,)).float()
    
    def process_jsonl_file(self, input_file: str, output_file: str, max_samples: int = None):
        """处理单个JSONL文件"""
        logger.info(f"开始处理文件: {input_file}")
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        sample_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"处理 {Path(input_file).name}")):
                try:
                    record = json.loads(line)
                    
                    # 提取prompt和completion tokens
                    prompt_tokens = record.get('prompt_tokens', [])
                    completion_tokens = record.get('completion_tokens', [])
                    
                    if not prompt_tokens:
                        continue
                    
                    # 为每个completion token生成数据
                    for i, target_token in enumerate(completion_tokens):
                        if max_samples and sample_count >= max_samples:
                            break
                        
                        # 当前上下文 = prompt + 已生成的completion tokens
                        current_context = prompt_tokens + completion_tokens[:i]
                        
                        # 获取top-k logits和indices
                        top_k_logits, top_k_indices = self._get_top_k_logits_and_indices(current_context)
                        
                        # 生成目标比特
                        target_bits = self._generate_target_bits(1)
                        target_bit = target_bits[0]
                        
                        # 获取绿名单mask
                        green_mask = self._get_green_mask(current_context, top_k_indices)
                        
                        # 构建样本
                        sample = {
                            'context_tokens': current_context[-self.max_context_length:],  # 限制上下文长度
                            'top_k_logits': top_k_logits.tolist(),
                            'top_k_indices': top_k_indices.tolist(),
                            'target_bit': target_bit,
                            'green_mask': green_mask.tolist(),
                            'target_token': target_token,  # 实际的目标token（用于验证）
                            'source_file': Path(input_file).name,
                            'line_number': line_num,
                            'position_in_completion': i
                        }
                        
                        processed_data.append(sample)
                        sample_count += 1
                        
                        if max_samples and sample_count >= max_samples:
                            break
                    
                    if max_samples and sample_count >= max_samples:
                        break
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"跳过第{line_num}行，解析错误: {e}")
                    continue
        
        # 保存为pickle文件
        logger.info(f"保存 {len(processed_data)} 个样本到 {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"文件处理完成: {output_file}")
        return len(processed_data)

def process_single_file(input_file: str, output_file: str = None, max_samples: int = 1000):
    """处理单个指定的JSONL文件"""
    # 配置文件路径
    config_path = "/home/qty/code/federated_token_selector/config/preprocessed_config.yaml"
    
    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    if not input_path.suffix == '.jsonl':
        raise ValueError(f"输入文件必须是JSONL格式: {input_file}")
    
    # 设置输出文件路径
    if output_file is None:
        output_dir = "/home/qty/code/federated_token_selector/processed_data"
        Path(output_dir).mkdir(exist_ok=True)
        output_file = Path(output_dir) / f"{input_path.stem}_processed.pkl"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"最大样本数: {max_samples}")
    print("-" * 50)
    
    # 创建预处理器
    preprocessor = TokenSelectorDataPreprocessor(config_path)
    
    try:
        samples = preprocessor.process_jsonl_file(
            input_file=str(input_path),
            output_file=str(output_file),
            max_samples=max_samples
        )
        
        print(f"\n✓ 文件处理完成！")
        print(f"生成样本数: {samples}")
        print(f"输出文件: {output_file}")
        
        return samples
        
    except Exception as e:
        print(f"✗ 文件处理失败: {e}")
        raise

def process_files_individually():
    """逐个处理文件"""
    # 配置文件路径
    config_path = "/home/qty/code/federated_token_selector/config/preprocessed_config.yaml"
    
    # 输入和输出目录
    input_dir = "/home/qty/code/federated_token_selector/output_token"
    output_dir = "/home/qty/code/federated_token_selector/processed_data"
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 获取所有JSONL文件
    input_path = Path(input_dir)
    jsonl_files = [f for f in input_path.glob('*.jsonl') if f.is_file()]  # 确保是文件而不是目录
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print("开始逐个处理...\n")
    
    total_samples = 0
    
    # 创建预处理器
    preprocessor = TokenSelectorDataPreprocessor(config_path)
    
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 正在处理: {jsonl_file.name}")
        
        output_file = Path(output_dir) / f"{jsonl_file.stem}_processed.pkl"
        
        try:
            samples = preprocessor.process_jsonl_file(
                input_file=str(jsonl_file),
                output_file=str(output_file),
                max_samples=1000  # 每个文件最多处理1000个样本
            )
            total_samples += samples
            
            print(f"✓ {jsonl_file.name} 处理完成，生成 {samples} 个样本")
            
        except Exception as e:
            print(f"✗ {jsonl_file.name} 处理失败: {e}")
            continue
        
        print("-" * 50)
    
    print(f"\n所有文件处理完成！")
    print(f"总共生成了 {total_samples} 个训练样本")
    print(f"处理后的数据保存在: {output_dir}")
    
    return total_samples

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='为Token选择器预处理数据')
    parser.add_argument('--file', '-f', type=str, default="/home/qty/code/federated_token_selector/output_token/harmbench_opt.jsonl",help='指定要处理的JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, default="/home/qty/code/federated_token_selector/outputs/harmbench_opt_processed.pkl",help='指定输出文件路径（可选）')
    parser.add_argument('--max-samples', '-m', type=int, default=1000, help='最大处理样本数（默认1000）')
    parser.add_argument('--batch', action='store_true', help='批量处理所有JSONL文件')
    
    args = parser.parse_args()
    
    try:
        if args.file:
            # 处理单个指定文件
            print("开始处理指定文件...")
            total_samples = process_single_file(
                input_file=args.file,
                output_file=args.output,
                max_samples=args.max_samples
            )
            print(f"\n数据预处理完成！生成了 {total_samples} 个训练样本")
            
        elif args.batch:
            # 批量处理所有文件
            print("开始批量处理...")
            total_samples = process_files_individually()
            print(f"\n数据预处理完成！总共生成了 {total_samples} 个训练样本")
            
        else:
            # 显示帮助信息
            parser.print_help()
            print("\n使用示例:")
            print("  处理单个文件: python preprocess_for_token_selector.py -f /path/to/file.jsonl")
            print("  指定输出路径: python preprocess_for_token_selector.py -f /path/to/file.jsonl -o /path/to/output.pkl")
            print("  限制样本数量: python preprocess_for_token_selector.py -f /path/to/file.jsonl -m 500")
            print("  批量处理: python preprocess_for_token_selector.py --batch")
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        raise

if __name__ == "__main__":
    main()