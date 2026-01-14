from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
from utils.logger import logger

class SentenceRetrievalModule:
    def __init__(self, model_path='./sentence_retrieval_pytorch_bert_base_model_FIXED'):
        """
        Loads the local Sentence Retrieval model.
        Assumes 'model_path' points to the FIXED version (resized embeddings).
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"⏳ Loading Sentence Retrieval Model from: {model_path}")
        
        try:
            # 1. Try loading the local FIXED model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
                
            self.model = SentenceTransformer(model_path, device=str(self.device))
            print(f"✅ Loaded LOCAL model successfully on {self.device}")
            
        except Exception as e:
            # 2. Fallback only if local fails
            print(f"⚠️ Failed to load local model ({e}). Downloading fallback...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=str(self.device))

        # Ensure max length is set
        if not hasattr(self.model, 'max_seq_length') or self.model.max_seq_length is None:
            self.model.max_seq_length = 512

    def retrieve_sentences(self, text, claim, top_k=5):
        """Retrieve most relevant sentences for a claim."""
        # Clean sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if not sentences:
            return []
        
        try:
            # Encode Claim
            claim_embedding = self.model.encode(
                [claim], 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
            
            # Encode Corpus
            sentence_embeddings = self.model.encode(
                sentences, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
            
            # Similarity
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Top-K
            # Ensure we don't ask for more top_k than we have sentences
            k = min(top_k, len(sentences))
            if k == 0: return []

            # Get indices of top scores (descending)
            top_indices = torch.topk(similarities, k).indices.cpu().numpy()
            
            results = []
            for i in top_indices:
                results.append((sentences[i], float(similarities[i])))
            
            return results

        except Exception as e:
            logger.error(f"Error during sentence retrieval: {e}")
            return []