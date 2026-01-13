"""
Sentence Retrieval Module for ProVe
Fixed: Forces online model loading to bypass local vocabulary mismatch errors.
"""
from sentence_transformers import SentenceTransformer, util, models
import torch
import numpy as np
import os
from utils.logger import logger

class SentenceRetrievalModule:
    def __init__(self, model_path='./sentence_retrieval_pytorch_bert_base_model_OG'):
        """
        Initialize Sentence Retrieval.
        CRITICAL FIX: We default to the online 'all-MiniLM-L6-v2' model.
        The local model at 'model_path' has a vocabulary mismatch (IndexError) and causes crashes.
        """
        # Define the device locally (CPU is safer for retrieval)
        self.device = torch.device("cpu")
        
        print(f"⚠️ DEBUG: Bypassing local model at {model_path} to prevent Vocabulary Mismatch crash.")
        print(f"✅ Loading standard 'all-MiniLM-L6-v2' from HuggingFace...")
        
        # Load standard stable model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        
        # Explicitly set max sequence length to avoid overflow
        self.model.max_seq_length = 512

    def retrieve_sentences(self, text, claim, top_k=5):
        """Retrieve most relevant sentences for a claim using CPU"""
        # Basic cleaning and splitting
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        try:
            # Encode on CPU
            # Note: We rely on the model's internal max_seq_length for truncation
            claim_embedding = self.model.encode(
                [claim], 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
            
            sentence_embeddings = self.model.encode(
                sentences, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
            
            # Compute cosine similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities.cpu().numpy())[-top_k:][::-1]
            results = [(sentences[i], float(similarities[i])) for i in top_indices]
            
            return results
        except Exception as e:
            logger.error(f"Error during sentence retrieval: {e}")
            return []