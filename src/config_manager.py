import yaml
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理器 - 用于加载和管理实验配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        """
        if self.config_path is None:
            # 使用默认配置
            default_config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
            self.config_path = str(default_config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键（如 'model.hidden_dim'）
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"配置键不存在: {key}")
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 导航到最后一级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        批量更新配置
        
        Args:
            updates: 更新字典，键支持点分隔的嵌套键
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，如果为None则覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"配置已保存到: {output_path}")
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        """
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        获取训练配置
        """
        return self.get('training', {})
    
    def get_federated_config(self) -> Dict[str, Any]:
        """
        获取联邦学习配置
        """
        return self.get('federated', {})
    
    def get_watermark_config(self) -> Dict[str, Any]:
        """
        获取水印配置
        """
        return self.get('watermark', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        获取数据配置
        """
        return self.get('data', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        获取评估配置
        """
        return self.get('evaluation', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        获取实验配置
        """
        return self.get('experiment', {})
    
    def create_training_config_dict(self) -> Dict[str, Any]:
        """
        创建用于训练的配置字典（兼容原有代码）
        """
        model_config = self.get_model_config()
        training_config = self.get_training_config()
        watermark_config = self.get_watermark_config()
        data_config = self.get_data_config()
        
        return {
            # 模型参数
            'vocab_size': model_config.get('vocab_size', 50000),
            'hidden_dim': model_config.get('hidden_dim', 256),
            'top_k': model_config.get('top_k', 50),
            
            # 水印参数
            'gamma': watermark_config.get('gamma', 0.25),
            
            # 训练参数
            'learning_rate': training_config.get('learning_rate', 1e-4),
            'batch_size': training_config.get('batch_size', 32),
            'samples_per_client': training_config.get('samples_per_client', 1000),
            'num_rounds': self.get('federated.num_rounds', 100),
            
            # 损失权重
            'lambda_watermark': training_config.get('loss_weights', {}).get('watermark', 1.0),
            'lambda_semantic': training_config.get('loss_weights', {}).get('semantic', 0.5),
            'lambda_fluency': training_config.get('loss_weights', {}).get('fluency', 0.3),
            
            # 数据参数
            'max_seq_len': data_config.get('max_seq_len', 128),
        }
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 检查必需的配置项
            required_keys = [
                'model.vocab_size',
                'model.hidden_dim',
                'model.top_k',
                'watermark.gamma',
                'training.learning_rate',
                'training.batch_size',
                'federated.num_clients',
                'federated.num_rounds'
            ]
            
            for key in required_keys:
                value = self.get(key)
                if value is None:
                    logger.error(f"缺少必需的配置项: {key}")
                    return False
            
            # 检查数值范围
            if self.get('watermark.gamma') <= 0 or self.get('watermark.gamma') >= 1:
                logger.error("watermark.gamma 必须在 (0, 1) 范围内")
                return False
            
            if self.get('training.learning_rate') <= 0:
                logger.error("training.learning_rate 必须大于 0")
                return False
            
            if self.get('federated.num_clients') <= 0:
                logger.error("federated.num_clients 必须大于 0")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def print_config(self) -> None:
        """
        打印当前配置
        """
        print("当前配置:")
        print(yaml.dump(self.config, default_flow_style=False, 
                       allow_unicode=True, indent=2))
    
    def create_experiment_variants(self, 
                                 param_grid: Dict[str, list]) -> list:
        """
        创建实验变体（用于超参数搜索）
        
        Args:
            param_grid: 参数网格，如 {'training.learning_rate': [1e-3, 1e-4], 'model.hidden_dim': [128, 256]}
            
        Returns:
            配置变体列表
        """
        import itertools
        
        # 获取所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        variants = []
        for combination in itertools.product(*values):
            # 创建新的配置
            variant_config = ConfigManager(self.config_path)
            
            # 更新参数
            updates = dict(zip(keys, combination))
            variant_config.update(updates)
            
            variants.append(variant_config)
        
        return variants

def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    return ConfigManager(config_path)

def create_config_from_dict(config_dict: Dict[str, Any], 
                           output_path: str) -> ConfigManager:
    """
    从字典创建配置文件
    
    Args:
        config_dict: 配置字典
        output_path: 输出路径
        
    Returns:
        配置管理器实例
    """
    # 保存配置到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, 
                 allow_unicode=True, indent=2)
    
    # 返回配置管理器
    return ConfigManager(output_path)

if __name__ == '__main__':
    # 测试配置管理器
    config = load_config()
    
    print("配置验证结果:", config.validate_config())
    
    print("\n模型配置:")
    print(config.get_model_config())
    
    print("\n训练配置字典:")
    print(config.create_training_config_dict())
    
    # 测试参数更新
    config.set('model.hidden_dim', 512)
    print(f"\n更新后的hidden_dim: {config.get('model.hidden_dim')}")
    
    # 测试实验变体创建
    param_grid = {
        'training.learning_rate': [1e-3, 1e-4],
        'model.hidden_dim': [128, 256]
    }
    
    variants = config.create_experiment_variants(param_grid)
    print(f"\n创建了 {len(variants)} 个实验变体")
    
    for i, variant in enumerate(variants):
        lr = variant.get('training.learning_rate')
        hidden_dim = variant.get('model.hidden_dim')
        print(f"变体 {i+1}: lr={lr}, hidden_dim={hidden_dim}")