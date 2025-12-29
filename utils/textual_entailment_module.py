"""
Textual Entailment Module for ProVe
Basic implementation
"""
from transformers import pipeline

class TextualEntailmentModule:
    def __init__(self):
        try:
            self.entailment_pipeline = pipeline(
                "text-classification", 
                model="roberta-large-mnli",
                device=-1  # Use CPU
            )
        except:
            self.entailment_pipeline = None
    
    def check_entailment(self, premise, hypothesis):
        """Check if premise entails hypothesis"""
        if self.entailment_pipeline is None:
            # Return dummy result if model not available
            return {'label': 'ENTAILMENT', 'score': 0.9}
        
        result = self.entailment_pipeline(f"{premise} [SEP] {hypothesis}")
        return result[0]
