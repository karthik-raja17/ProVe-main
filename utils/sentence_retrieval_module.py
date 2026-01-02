"""
Sentence Retrieval Module for ProVe
Updated to fix vocabulary size mismatch and use hard-coded DEVICE
"""
from sentence_transformers import SentenceTransformer, util, models
import torch
import numpy as np
import os
from utils.logger import logger

# Hard-coded for the Eurecom cluster allocation (GPU index 0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Sentence Retrieval Module initialized on {DEVICE}")

class SentenceRetrievalModule:
    def __init__(self, model_path='/home/kandavel/ProVe-main/sentence_retrieval_pytorch_bert_base_model_OG'):
        """
        Initialize with local BERT model weights by manually defining modules
        to avoid vocabulary size mismatch (28996 vs 30522).
        """
        if os.path.exists(model_path):
            print(f"Connecting local Sentence Retrieval model from: {model_path}")
            
            # 1. Manually load the transformer layer to use local config.json and vocab.txt
            # This prevents the 'size mismatch for weight' error
            word_embedding_model = models.Transformer(model_path, max_seq_length=512)
            
            # 2. Add the Pooling layer (Mean Pooling is required for valid embeddings)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            # 3. Assemble the modules into a SentenceTransformer
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)
        else:
            print(f"Warning: Local path {model_path} not found. Falling back to online model.")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    def retrieve_sentences(self, text, claim, top_k=5):
        """Retrieve most relevant sentences for a claim using local BERT embeddings"""
        # Split text into sentences
        # Note: If NLTK is available in your environment, consider nltk.sent_tokenize(text)
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        try:
            # Encode claim and sentences on the assigned GPU
            # convert_to_tensor=True ensures calculation happens in VRAM
            claim_embedding = self.model.encode([claim], convert_to_tensor=True)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k sentences
            top_indices = np.argsort(similarities.cpu())[-top_k:][::-1]
            results = [(sentences[i], float(similarities[i])) for i in top_indices]
            
            return results
        except Exception as e:
            print(f"Error during sentence retrieval inference: {e}")
            return []