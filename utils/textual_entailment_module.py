"""
Textual Entailment Module for ProVe
Fixed: 
1. Added mapping for 'LABEL_X' outputs to 'SUPPORTS/REFUTES/NEI'.
2. Fixed get_batch_scores to return clean dictionary keys.
"""
import torch
from transformers import pipeline

class TextualEntailmentModule():
    def __init__(
        self,
        # Ensure these paths point to your local model folder
        model_path='./textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/',
        tokenizer_path='./textual_entailment_BERT_FEVER_v4_PBT_OG/BERT_FEVER_v4_model_PBT/'
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

        # MAPPING: Adjust these if your specific model uses a different order.
        # Common FEVER/VitaminC convention: 0=SUPPORTS, 1=REFUTES, 2=NEI
        self.id2label = {
            'LABEL_0': 'SUPPORTS',
            'LABEL_1': 'REFUTES',
            'LABEL_2': 'NOT_ENOUGH_INFO' 
        }

    def get_batch_scores(self, claims: list, evidence: list):
        """
        Calculates entailment scores for a batch of evidence/claim pairs.
        Returns: List of dicts like {'SUPPORTS': 0.95, 'REFUTES': 0.05, ...}
        """
        # 1. Format for BERT: "[CLS] claim [SEP] evidence"
        formatted_inputs = [f"{h} [SEP] {p}" for p, h in zip(evidence, claims)]
        
        try:
            # 2. Run Batch Inference
            results = self.entailment_pipeline(formatted_inputs, batch_size=16)
            
            # 3. Format Output
            batch_output = []
            for res in results:
                # Map 'LABEL_X' to readable strings
                prob_map = {}
                for item in res:
                    label_raw = item['label']
                    score = item['score']
                    
                    # Convert LABEL_0 -> SUPPORTS
                    clean_label = self.id2label.get(label_raw, label_raw).upper()
                    
                    # Normalize 'NEI' variations to 'NOT_ENOUGH_INFO'
                    if clean_label == 'NEI': 
                        clean_label = 'NOT_ENOUGH_INFO'
                        
                    prob_map[clean_label] = score
                
                batch_output.append(prob_map)
                
            return batch_output
            
        except Exception as e:
            print(f"Error in batch entailment: {e}")
            # Return neutral distribution on error
            return [{'SUPPORTS': 0.33, 'REFUTES': 0.33, 'NOT_ENOUGH_INFO': 0.33}] * len(claims)

    def get_label_from_scores(self, scores: dict):
        """
        Helper method to return the label with the highest score.
        """
        if not scores:
            return "NOT_ENOUGH_INFO"
        return max(scores, key=scores.get)

    def check_entailment(self, premise, hypothesis):
        """Legacy method for backward compatibility"""
        results = self.get_batch_scores(evidence=[premise], claims=[hypothesis])
        return results[0]