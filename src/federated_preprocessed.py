import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import os

from config_manager import ConfigManager
from token_selector import TokenSelectorController, WatermarkBitGenerator, GreenListGenerator
from federated_framework import FederatedClient, FederatedServer, FedAvgAggregator
from llm_interface import LLMInterfaceFactory
from watermark_detector import WatermarkDetector

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 预处理数据加载器 ---
class PreprocessedTokenDataset(Dataset):
    def __init__(self, data_path, samples_per_client=None):
        self.data_path = data_path
        self.samples_per_client = samples_per_client
        self.data = self._load_data()
        
    def _load_data(self):
        data = []
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} does not exist")
            return data
            
        try:
            if self.data_path.endswith('.pkl'):
                # 加载pickle文件
                with open(self.data_path, 'rb') as f:
                    raw_data = pickle.load(f)
                    if isinstance(raw_data, list):
                        data = raw_data
                    else:
                        print(f"Warning: Pickle file contains {type(raw_data)}, expected list")
                        return data
            elif self.data_path.endswith('.jsonl'):
                # 加载JSONL文件并转换格式
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            item = json.loads(line.strip())
                            # 检查是否是新格式（预处理后的数据）
                            if all(key in item for key in ['context_tokens', 'top_p_logits', 'top_p_indices', 'target_bit']):
                                data.append(item)
                            # 检查是否是旧格式（原始数据）
                            elif all(key in item for key in ['prompt_tokens', 'completion_tokens']):
                                # 转换为预处理格式（这里需要实际的预处理逻辑）
                                print(f"Warning: Found old format data in {self.data_path}. This data needs to be preprocessed first.")
                                # 暂时跳过，因为需要真正的预处理
                                continue
                            else:
                                print(f"Warning: Unknown data format in line {line_num + 1}")
                                continue
                                
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                            continue
                        except Exception as e:
                            print(f"Warning: Error processing line {line_num + 1}: {e}")
                            continue
            else:
                print(f"Warning: Unsupported file format: {self.data_path}")
                return data
                
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            return data
            
        # 限制样本数量
        if self.samples_per_client and len(data) > self.samples_per_client:
            data = data[:self.samples_per_client]
            
        print(f"Loaded {len(data)} samples from {self.data_path}")
        return data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'context_tokens': torch.tensor(item['context_tokens'], dtype=torch.long),
            'top_p_logits': torch.tensor(item['top_p_logits'], dtype=torch.float32),
            'top_p_indices': torch.tensor(item['top_p_indices'], dtype=torch.long),
            'target_bit': torch.tensor(item['target_bit'], dtype=torch.float32)
        }

# --- 联邦学习客户端 ---
class PreprocessedClient(FederatedClient):
    """使用预处理数据的联邦学习客户端"""
    def train(self, global_model_state: dict):
        self.model.load_state_dict(global_model_state)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('training', {}).get('learning_rate', 1e-4))
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.config.get('training', {}).get('local_epochs', 1)):
            for batch in self.local_data:
                optimizer.zero_grad()
                output = self.model(batch['context_tokens'].to(self.device), 
                                    batch['top_p_logits'].to(self.device), 
                                    batch['top_p_indices'].to(self.device))
                loss = criterion(output, batch['target_bit'].to(self.device))
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict(), len(self.local_data.dataset)

# --- 水印文本生成器 ---
class WatermarkTextGenerator:
    """带水印的文本生成器"""
    
    def __init__(self, token_selector: TokenSelectorController, vocab_size: int, device: str = 'cpu', model_config: dict = None):
        self.token_selector = token_selector.to(device)
        self.vocab_size = vocab_size
        self.device = device
        
        # 创建LLM接口用于文本解码
        try:
            if model_config:
                self.llm_interface = LLMInterfaceFactory.create_llm_interface(
                    model_config.get('interface_type', 'huggingface'),
                    model_name=model_config.get('model_name', 'gpt2-xl'),
                    device=device,
                    cache_dir=model_config.get('cache_dir'),
                    local_files_only=True
                )
                logger.info(f"文本生成器使用模型: {model_config.get('model_name')}, 设备: {device}")
            else:
                # 默认配置
                self.llm_interface = LLMInterfaceFactory.create_llm_interface(
                    'huggingface',
                    model_name='gpt2-xl',
                    device=device,
                    cache_dir='/home/qty/code/gpt2-xl',
                    local_files_only=True
                )
                logger.info(f"文本生成器使用默认模型: gpt2-xl, 设备: {device}")
        except Exception as e:
            logger.warning(f"LLM接口创建失败: {e}，将使用模拟解码")
            self.llm_interface = None
        
        # 水印比特生成器
        self.bit_generator = WatermarkBitGenerator()
        
        # 绿名单生成器
        self.green_generator = GreenListGenerator(vocab_size=vocab_size)
        
        self.token_selector.eval()
    
    def decode_tokens(self, tokens: list) -> str:
        """解码token序列为文本"""
        if self.llm_interface:
            return self.llm_interface.decode(tokens)
        else:
            # 简单的模拟解码
            return f"[模拟文本: {len(tokens)} tokens]"
    
    def generate_text_with_watermark(self, prompt_tokens: list, max_length: int = 30, watermark_bits: list = None):
        """生成带水印的文本"""
        if watermark_bits is None:
            watermark_bits = self.bit_generator.generate_bits(max_length)
        
        generated_tokens = prompt_tokens.copy()
        embedded_bits = []
        
        for i in range(max_length):
            if i >= len(watermark_bits):
                break
                
            # 模拟生成过程（简化版）
            context = torch.tensor(generated_tokens[-10:]).unsqueeze(0).to(self.device)  # 使用最后10个token作为上下文
            
            # 生成候选token（模拟）
            candidate_tokens = list(range(min(50, self.vocab_size)))  # 简化的候选集
            logits = torch.randn(len(candidate_tokens)).to(self.device)  # 模拟logits
            indices = torch.tensor(candidate_tokens).to(self.device)
            
            # 使用token选择器
            with torch.no_grad():
                try:
                    # 调整输入维度以匹配模型期望
                    context_padded = torch.zeros(1, 256).to(self.device)  # 假设模型期望256维上下文
                    context_len = min(len(generated_tokens), 256)
                    if context_len > 0:
                        context_padded[0, :context_len] = torch.tensor(generated_tokens[-context_len:]).to(self.device)
                    
                    logits_input = logits.unsqueeze(0)  # [1, num_candidates]
                    indices_input = indices.unsqueeze(0)  # [1, num_candidates]
                    
                    # 确保输入维度正确
                    if logits_input.size(1) < 50:  # 如果候选数量不足，填充
                        pad_size = 50 - logits_input.size(1)
                        logits_input = torch.cat([logits_input, torch.zeros(1, pad_size).to(self.device)], dim=1)
                        indices_input = torch.cat([indices_input, torch.zeros(1, pad_size, dtype=torch.long).to(self.device)], dim=1)
                    
                    output = self.token_selector(context_padded, logits_input, indices_input)
                    probabilities = torch.softmax(output, dim=-1)
                    
                    # 根据水印比特选择token
                    target_bit = watermark_bits[i]
                    greenlist_ids = self.green_generator.get_greenlist_ids(torch.tensor(generated_tokens))
                    
                    if target_bit == 1:
                        # 选择绿名单中的token
                        green_candidates = [idx for idx in candidate_tokens if idx in greenlist_ids]
                        if green_candidates:
                            selected_token = green_candidates[0]  # 简化选择
                        else:
                            selected_token = candidate_tokens[0]
                    else:
                        # 选择非绿名单中的token
                        non_green_candidates = [idx for idx in candidate_tokens if idx not in greenlist_ids]
                        if non_green_candidates:
                            selected_token = non_green_candidates[0]  # 简化选择
                        else:
                            selected_token = candidate_tokens[0]
                    
                    generated_tokens.append(selected_token)
                    embedded_bits.append(target_bit)
                    
                except Exception as e:
                    logger.warning(f"Token选择器调用失败: {e}，使用随机选择")
                    selected_token = candidate_tokens[0]
                    generated_tokens.append(selected_token)
                    embedded_bits.append(watermark_bits[i])
        
        return generated_tokens, embedded_bits
    
    def generate_text_without_watermark(self, prompt_tokens: list, max_length: int = 30):
        """生成不带水印的文本"""
        generated_tokens = prompt_tokens.copy()
        
        for i in range(max_length):
            # 简单的随机生成
            next_token = torch.randint(0, min(1000, self.vocab_size), (1,)).item()
            generated_tokens.append(next_token)
        
        return generated_tokens

# --- 主训练函数 ---
def run_federated_training(config_path: str):
    """运行基于预处理数据的联邦训练"""
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    device = config.get('experiment', {}).get('device', 'cpu')

    # 初始化全局模型
    global_model = TokenSelectorController(
        vocab_size=config.get('model', {}).get('vocab_size'),
        context_dim=config.get('model', {}).get('hidden_dim'),
        hidden_dim=config.get('model', {}).get('hidden_dim'),
        top_k=config.get('model', {}).get('max_candidate_tokens')
    ).to(device)

    # 创建客户端
    clients = []
    for client_config in config.get('federated', {}).get('clients', []):
        dataset = PreprocessedTokenDataset(client_config['data_path'], config)
        dataloader = DataLoader(dataset, batch_size=config.get('training', {}).get('batch_size', 32), shuffle=True)
        client = PreprocessedClient(
            client_id=client_config['client_id'],
            model=TokenSelectorController(
                vocab_size=config.get('model', {}).get('vocab_size'),
                context_dim=config.get('model', {}).get('hidden_dim'),
                hidden_dim=config.get('model', {}).get('hidden_dim'),
                top_k=config.get('model', {}).get('max_candidate_tokens')
            ).to(device),
            local_data=dataloader,
            device=device,
            config=config
        )
        clients.append(client)

    # 创建服务器
    server = FederatedServer(
        clients=clients,
        global_model=global_model,
        aggregator=FedAvgAggregator(),
        config=config
    )

    # 创建文本生成器和水印检测器
    text_generator = WatermarkTextGenerator(
        token_selector=global_model,
        vocab_size=config.get('model', {}).get('vocab_size'),
        device=device
    )
    
    watermark_detector = WatermarkDetector(
        vocab_size=config.get('model', {}).get('vocab_size'),
        gamma=config.get('watermark', {}).get('gamma', 0.25),
        hash_key=config.get('watermark', {}).get('hash_key', 15485863)
    )

    # 运行联邦训练
    output_dir = config.get('experiment', {}).get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 准备测试提示
    test_prompts = [
        [2, 23742, 4128, 3154, 8, 19, 171, 1434, 11007, 154, 49, 2349, 6, 1712, 5, 1374, 458, 9, 3742, 1703, 634, 804, 4927, 518, 11, 5, 4, 104, 4, 37542, 10193, 2246, 54, 1368, 705, 819, 743, 565, 8, 45, 28, 16997, 98, 2636, 785, 25, 24, 13458, 819, 37542, 10193, 3303, 5314, 5394, 4, 509, 266, 11, 436, 1681, 3649, 14, 4927, 13, 1111, 2737, 390, 16, 22, 32278, 28075, 113, 8, 22, 90, 5556, 173, 113, 8, 12297, 86, 409, 31, 5286, 19026, 6, 8, 2127, 390, 11, 10, 27180, 737, 9, 519, 7, 2394, 1081, 1282, 136, 2065, 1111, 4158, 4, 5365, 26, 98, 6, 114, 11540, 4554, 16, 8698, 745, 3069, 1717, 5658, 45, 1368, 705, 1009, 55, 20435, 683, 10, 186, 25, 24, 40, 1888, 1296, 1389, 11, 809, 8, 12113, 9499, 1175, 40, 28, 1202, 4, 35968, 11, 369, 1327, 16, 15351, 626, 223, 284, 13702, 6, 2333, 11, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [100, 200, 300, 400, 500]
    ]

    for round_num in range(1, config.get('federated', {}).get('num_rounds', 1) + 1):
        logger.info(f"--- Round {round_num} ---")
        server.train_round()
        
        # 每轮训练后生成和输出文本
        if round_num % 5 == 0:  # 每5轮输出一次
            logger.info(f"\n=== Round {round_num} 文本生成结果 ===")
            
            for i, prompt_tokens in enumerate(test_prompts[:1]):  # 只使用第一个提示以节省输出
                # 生成带水印的文本
                watermarked_tokens, embedded_bits = text_generator.generate_text_with_watermark(
                    prompt_tokens=prompt_tokens,
                    max_length=20
                )
                
                # 生成不带水印的文本
                unwatermarked_tokens = text_generator.generate_text_without_watermark(
                    prompt_tokens=prompt_tokens,
                    max_length=20
                )
                
                # 解码为文本
                watermarked_text = text_generator.decode_tokens(watermarked_tokens)
                unwatermarked_text = text_generator.decode_tokens(unwatermarked_tokens)
                prompt_text = text_generator.decode_tokens(prompt_tokens)
                
                # 检测水印
                detection_result = watermark_detector.detect_watermark(watermarked_tokens)
                
                logger.info(f"\n提示文本: {prompt_text}")
                logger.info(f"水印文本: {watermarked_text}")
                logger.info(f"原始文本: {unwatermarked_text}")
                logger.info(f"嵌入比特: {embedded_bits[:10]}...")
                logger.info(f"水印检测: {'检测到水印' if detection_result['is_watermarked'] else '未检测到水印'} (z-score: {detection_result['z_score']:.2f})")
        
        # 保存模型
        if round_num % 10 == 0:
            model_path = os.path.join(output_dir, f'model_round_{round_num}.pth')
            torch.save(server.global_model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

    logger.info("Federated training completed.")

# --- 主程序入口 ---
if __name__ == '__main__':
    config_file = Path(__file__).parent.parent / 'config' / 'preprocessed_config.yaml'
    run_federated_training(str(config_file))