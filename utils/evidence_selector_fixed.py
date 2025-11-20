"""
Fixed Evidence Selector for ProVe
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np

class EvidenceSelector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def select_evidence(self, claim, sentences, top_k=5):
        """Select most relevant sentences for a claim"""
        if not sentences:
            return []
        
        try:
            # Encode claim and sentences
            claim_embedding = self.model.encode([claim], convert_to_tensor=True)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Compute similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k sentences - FIXED: Handle case where top_k > available sentences
            available_sentences = min(top_k, len(sentences))
            if available_sentences == 0:
                return []
                
            top_indices = np.argsort(similarities.cpu())[-available_sentences:][::-1]
            evidence = [(sentences[i], float(similarities[i])) for i in top_indices]
            
            return evidence
        except Exception as e:
            print(f"Error selecting evidence: {e}")
            return []
