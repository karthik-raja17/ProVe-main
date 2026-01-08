"""
Textual Entailment Module for ProVe
Fixed: 
1. Added get_label_from_scores helper (Fixes AttributeError).
2. Retained batch processing and correct argument names.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np

class TextualEntailmentModule():
    def __init__(
        self,
        # Ensure these paths point to your local model folder
        model_path='/home/kandavel/ProVe-main/textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/',
        tokenizer_path='/home/kandavel/ProVe-main/textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/'
    ):
        self.device_id = 0 if torch.cuda.is_available() else -1
        
        print(f"✅ Initializing Entailment Pipeline on {'GPU' if self.device_id == 0 else 'CPU'}...")
        
        # Initialize the pipeline
        self.entailment_pipeline = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=tokenizer_path,
            device=self.device_id,
            top_k=None 
        )

    def get_batch_scores(self, claims: list, evidence: list):
        """
        Calculates entailment scores for a batch of evidence/claim pairs.
        Args:
            claims (list): The list of Verbalized Claims (Hypotheses)
            evidence (list): The list of Evidence strings (Premises)
        """
        # 1. Map arguments to NLI concepts
        premises = evidence
        hypotheses = claims
        
        # 2. Format for BERT: "[CLS] claim [SEP] evidence"
        formatted_inputs = [f"{h} [SEP] {p}" for p, h in zip(premises, hypotheses)]
        
        try:
            # 3. Run Batch Inference
            results = self.entailment_pipeline(formatted_inputs, batch_size=16)
            
            # 4. Format Output
            batch_output = []
            for res in results:
                # Convert list of dicts to single dict {'SUPPORTS': 0.9, ...}
                prob_map = {item['label'].upper(): item['score'] for item in res}
                batch_output.append(prob_map)
                
            return batch_output
            
        except Exception as e:
            print(f"Error in batch entailment: {e}")
            return [{'SUPPORTS': 0.33, 'REFUTES': 0.33, 'NOT_ENOUGH_INFO': 0.33}] * len(claims)

    def get_label_from_scores(self, scores: dict):
        """
        Helper method to return the label with the highest score.
        Input: {'SUPPORTS': 0.9, 'REFUTES': 0.05, 'NEI': 0.05}
        Output: 'SUPPORTS'
        """
        if not scores:
            return "NOT_ENOUGH_INFO"
        # Return the key corresponding to the max value
        return max(scores, key=scores.get)

    def check_entailment(self, premise, hypothesis):
        """Legacy method for backward compatibility"""
        # Map single call to batch function
        results = self.get_batch_scores(evidence=[premise], claims=[hypothesis])
        return results[0]