"""
Sentence Retrieval Module for ProVe
Basic implementation
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SentenceRetrievalModule:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def retrieve_sentences(self, text, claim, top_k=5):
        """Retrieve most relevant sentences for a claim"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        # Encode claim and sentences
        claim_embedding = self.model.encode([claim], convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Compute similarity
        similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
        
        # Get top-k sentences
        top_indices = np.argsort(similarities.cpu())[-top_k:][::-1]
        results = [(sentences[i], float(similarities[i])) for i in top_indices]
        
        return results
