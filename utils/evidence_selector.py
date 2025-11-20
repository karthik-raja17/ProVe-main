"""
Perfect Evidence Selector for ProVe
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
            # Filter out empty sentences
            valid_sentences = [s for s in sentences if s and len(s.strip()) > 10]
            if not valid_sentences:
                return []
            
            # Encode claim and sentences
            claim_embedding = self.model.encode([claim], convert_to_tensor=True)
            sentence_embeddings = self.model.encode(valid_sentences, convert_to_tensor=True)
            
            # Compute similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k sentences - handle edge cases
            available_sentences = min(top_k, len(valid_sentences))
            if available_sentences <= 0:
                return []
            
            # Use stable sorting to avoid "step must be greater than zero"
            similarity_scores = similarities.cpu().numpy()
            top_indices = np.argpartition(similarity_scores, -available_sentences)[-available_sentences:]
            top_indices = top_indices[np.argsort(similarity_scores[top_indices])][::-1]
            
            evidence = [(valid_sentences[i], float(similarity_scores[i])) for i in top_indices]
            return evidence
            
        except Exception as e:
            print(f"Evidence selection error: {e}")
            return []
