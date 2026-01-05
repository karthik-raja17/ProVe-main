"""
Sentence Retrieval Module for ProVe
Fixed: Vocabulary size mismatch, NameErrors, and scope issues
"""
from sentence_transformers import SentenceTransformer, util, models
import torch
import numpy as np
import os
from utils.logger import logger

class SentenceRetrievalModule:
    def __init__(self, model_path='/home/kandavel/ProVe-main/sentence_retrieval_pytorch_bert_base_model_OG'):
        """
        Initialize with local BERT model weights.
        Forced to CPU to avoid CUDA 'device-side assert' errors.
        """
        # Define the device locally
        self.device = torch.device("cpu")
        
        if os.path.exists(model_path):
            print(f"Connecting local Sentence Retrieval model from: {model_path}")
            
            # 1. Load transformer layer with local vocab/config
            word_embedding_model = models.Transformer(model_path, max_seq_length=512)
            
            # 2. Add Mean Pooling
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            # 3. Assemble onto CPU
            self.model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], 
                device="cpu"
            )
        else:
            print(f"Warning: Local path {model_path} not found. Falling back to online model.")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    def retrieve_sentences(self, text, claim, top_k=5):
        """Retrieve most relevant sentences for a claim using CPU"""
        # Basic cleaning and splitting
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        try:
            # Encode on CPU (convert_to_tensor=True for utility compatibility)
            claim_embedding = self.model.encode([claim], convert_to_tensor=True, device=self.device)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            
            # Compute cosine similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities.cpu().numpy())[-top_k:][::-1]
            results = [(sentences[i], float(similarities[i])) for i in top_indices]
            
            return results
        except Exception as e:
            logger.error(f"Error during sentence retrieval: {e}")
            return []