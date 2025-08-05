import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bert_score import score
import logging
from typing import List

logger = logging.getLogger(__name__)

class TextQualityEvaluator:
    """
    文本质量评估器，用于计算困惑度（PPL）和语义相似度（BERTScore）
    """
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器，加载所需的模型
        Args:
            device: 计算设备
        """
        self.device = device
        
        # 加载用于PPL计算的模型和分词器
        try:
            self.ppl_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            logger.info("成功加载GPT-2模型用于PPL计算。")
        except Exception as e:
            logger.error(f"加载GPT-2模型失败: {e}")
            self.ppl_model = None
            self.ppl_tokenizer = None

    def calculate_ppl(self, texts: List[str]) -> float:
        """
        计算给定文本列表的平均困惑度（PPL）
        """
        if not self.ppl_model or not self.ppl_tokenizer:
            logger.warning("PPL模型未加载，无法计算PPL。")
            return float('nan')
        
        total_neg_log_likelihood = 0
        total_tokens = 0
        
        for text in texts:
            if not text:
                continue
            
            encodings = self.ppl_tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(self.device)
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * input_ids.size(1)
            
            total_neg_log_likelihood += neg_log_likelihood.item()
            total_tokens += input_ids.size(1)
            
        if total_tokens == 0:
            return float('nan')
            
        ppl = torch.exp(torch.tensor(total_neg_log_likelihood / total_tokens))
        return ppl.item()

    def calculate_bert_score(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算候选文本和参考文本之间的BERTScore
        """
        if not candidates or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        try:
            P, R, F1 = score(candidates, references, lang='en', verbose=False, device=self.device)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"计算BERTScore失败: {e}")
            return {'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}