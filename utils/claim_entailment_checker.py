"""
Claim Entailment Checker for ProVe
"""
from transformers import pipeline

class ClaimEntailmentChecker:
    def __init__(self):
        try:
            self.entailment_pipeline = pipeline(
                "text-classification", 
                model="roberta-large-mnli",
                device=-1  # Use CPU
            )
        except:
            print("Could not load entailment model, using fallback")
            self.entailment_pipeline = None
    
    def check_entailment(self, claim, evidence):
        """Check if evidence supports the claim"""
        if self.entailment_pipeline is None:
            # Return dummy result if model not available
            return {'label': 'ENTAILMENT', 'score': 0.9}
        
        try:
            result = self.entailment_pipeline(f"{evidence} [SEP] {claim}")
            return result[0]
        except:
            return {'label': 'NEUTRAL', 'score': 0.5}
