"""
Fixed HTML to Evidence processor:
1. Replaced unstable np.argsort with torch.topk to fix 'step must be greater than zero' error.
2. Corrected max_seq_length to 256 (MiniLM limit).
"""
import nltk
import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from utils.logger import logger
from sentence_transformers import util
import torch

# Set NLTK data path
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except LookupError:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt_tab', paths=[nltk_data_dir])
except LookupError:
    nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_dir)

class HTMLSentenceProcessor:
    def __init__(self):
        # Set NLTK data path
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
        
    def html_to_text(self, html_content):
        """Extract clean text from HTML"""
        try:
            if not html_content:
                return ""
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error processing HTML: {e}")
            return ""
    
    def text_to_sentences(self, text):
        """Split text and HARD LIMIT character count to prevent overflow"""
        try:
            if not text:
                return []
            
            sentences = nltk.tokenize.sent_tokenize(text)
            
            valid_sentences = []
            for s in sentences:
                clean_s = s.strip()
                if len(clean_s) < 20:
                    continue
                
                # Hard limit 500 chars (~100-150 words)
                if len(clean_s) > 500:
                    valid_sentences.append(clean_s[:500])
                else:
                    valid_sentences.append(clean_s)
                    
            return valid_sentences
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []
    
    def process_html_to_sentences(self, html_df):
        """Process HTML DataFrame and extract sentences"""
        sentences_data = []
        for idx, row in html_df.iterrows():
            if row['status'] == 200 and row.get('html') is not None:
                text_content = self.html_to_text(row["html"])
                sentences = self.text_to_sentences(text_content)
                for sent in sentences:
                    sentences_data.append({
                        "url": row["url"],
                        "sentence": sent,
                        "html": row["html"],
                        "reference_id": row.get("reference_id", idx)
                    })
        return pd.DataFrame(sentences_data) if sentences_data else pd.DataFrame()

class EvidenceSelector:
    def __init__(self, sentence_retrieval=None, verb_module=None, model_name='all-MiniLM-L6-v2'):
        # 1. Load the model
        if sentence_retrieval is not None:
            self.model = sentence_retrieval
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        
        # 2. Identify the internal SentenceTransformer object
        self.actual_transformer = self.model.model if hasattr(self.model, 'model') else self.model
        
        # 3. SET Truncation via Attribute
        # MiniLM usually supports 256 tokens max
        if hasattr(self.actual_transformer, 'max_seq_length'):
            self.actual_transformer.max_seq_length = 256
        
        # 4. FORCE CPU for stability
        self.device = torch.device("cpu")
        self.actual_transformer.to(self.device)
        
        logger.info(f"Retriever initialized on {self.device} with max_seq_length: {getattr(self.actual_transformer, 'max_seq_length', 'Default')}")
        self.verb_module = verb_module
    
    def select_relevant_sentences(self, verbalized_claims, sentences_df, top_k=5):
        """Select most relevant sentences for each verbalized claim using torch.topk"""
        if sentences_df is None or sentences_df.empty:
            return pd.DataFrame()

        # 1. Normalize input to a list of dictionaries
        claims_to_process = []
        if isinstance(verbalized_claims, str):
            claims_to_process = [{'verbalized_claim': verbalized_claims}]
        elif isinstance(verbalized_claims, list):
            if not verbalized_claims: return pd.DataFrame()
            claims_to_process = [{'verbalized_claim': c} for c in verbalized_claims]
        elif isinstance(verbalized_claims, pd.DataFrame):
            if verbalized_claims.empty: return pd.DataFrame()
            claims_to_process = verbalized_claims.to_dict(orient='records')
        elif isinstance(verbalized_claims, pd.Series):
            claims_to_process = [{'verbalized_claim': c} for c in verbalized_claims.tolist()]
        else:
            logger.error(f"Unsupported type for verbalized_claims: {type(verbalized_claims)}")
            return pd.DataFrame()

        evidence_data = []
        
        # Get all sentences from the HTML pool
        sentences = sentences_df['sentence'].tolist()
        if not sentences:
            return pd.DataFrame()
        
        # 2. Encode all pool sentences
        try:
            sentence_embeddings = self.model.model.encode(
                sentences, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error encoding sentence pool: {e}")
            return pd.DataFrame()
        
        # 3. Process each claim
        for claim_dict in claims_to_process:
            claim_text = claim_dict.get('verbalized_claim', '')
            if not claim_text:
                continue
            
            try:
                claim_embedding = self.model.model.encode(
                    [claim_text], 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device
                )
                
                # Compute similarity
                similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
                
                # FIX: Use torch.topk instead of np.argsort slicing
                # This prevents "step must be greater than zero" error
                k = min(top_k, len(sentences))
                if k == 0: continue
                
                # Get scores and indices directly from PyTorch
                scores, indices = torch.topk(similarities, k=k)
                
                # Create evidence entries
                for score, idx in zip(scores, indices):
                    i = idx.item() # Convert tensor index to int
                    similarity_score = score.item()
                    
                    entry = {
                        'verbalized_claim': claim_text,
                        'sentence': sentences[i],
                        'similarity_score': similarity_score,
                        'url': sentences_df.iloc[i]['url'] if 'url' in sentences_df.columns else '',
                        'reference_id': sentences_df.iloc[i].get('reference_id', '')
                    }
                    
                    # Map metadata
                    metadata_keys = [
                        'claim_id', 'entity_id', 'property_id', 'object_id', 
                        'entity_label', 'property_label', 'object_label'
                    ]
                    for key in metadata_keys:
                        if key in claim_dict:
                            entry[key] = claim_dict[key]
                    
                    evidence_data.append(entry)
                    
            except Exception as e:
                logger.error(f"Error processing claim encoding: {e}")
                continue
        
        return pd.DataFrame(evidence_data) if evidence_data else pd.DataFrame()
    
    def select_evidence(self, claim, sentences, top_k=5):
        """Legacy method for single claim evidence selection"""
        if not sentences:
            return []
        
        try:
            claim_embedding = self.model.encode([claim], convert_to_tensor=True, device=self.device)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Use torch.topk here as well
            k = min(top_k, len(sentences))
            scores, indices = torch.topk(similarities, k=k)
            
            evidence = []
            for score, idx in zip(scores, indices):
                evidence.append((sentences[idx.item()], score.item()))
            
            return evidence
        except Exception as e:
            logger.error(f"Error selecting evidence: {e}")
            return []

# Backward compatibility function
def process_evidence(sentences_df, parser_result):
    selector = EvidenceSelector()
    verbalized_claims = parser_result['claims'].copy()
    verbalized_claims['verbalized_claim'] = verbalized_claims.apply(
        lambda row: f"{row['entity_label']} {row['property_label']} {row['object_label']}", 
        axis=1
    )
    return selector.select_relevant_sentences(verbalized_claims, sentences_df)