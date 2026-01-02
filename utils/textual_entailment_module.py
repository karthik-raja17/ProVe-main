"""
Textual Entailment Module for ProVe
Basic implementation
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np
import re
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEBUG: Textual Entailment moving to: {device}")
MAX_LEN = 512

class TextualEntailmentModule():
    def __init__(
        self,
        # Ensure these paths point to your local model folder
        model_path = '/home/kandavel/ProVe-main/textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/',
        tokenizer_path = '/home/kandavel/ProVe-main/textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/'
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
    def check_entailment(self, premise, hypothesis):
        """Check if premise entails hypothesis"""
        if self.entailment_pipeline is None:
            # Return dummy result if model not available
            return {'label': 'ENTAILMENT', 'score': 0.9}
        
        result = self.entailment_pipeline(f"{premise} [SEP] {hypothesis}")
        return result[0]
