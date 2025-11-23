"""
Fixed HTML to Evidence processor with correct NLTK path
"""
import nltk
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from utils.logger import logger

# Set NLTK data path
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except LookupError:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)

class HTMLSentenceProcessor:
    def __init__(self):
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
    def html_to_text(self, html_content):
        """Extract clean text from HTML"""
        try:
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
        """Split text into sentences using NLTK"""
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return sentences
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {e}")
            return []
    
    def process_html_to_sentences(self, html_df):
        """Process HTML DataFrame and extract sentences"""
        sentences_data = []
        for idx, row in html_df.iterrows():
            if row['status'] == 200 and row['html'] is not None:
                # Convert HTML to text
                text_content = self.html_to_text(row["html"])
                # Convert text to sentences
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
        # Use provided sentence retrieval model or create new one
        if sentence_retrieval is not None:
            self.model = sentence_retrieval
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        
        self.verb_module = verb_module
    
    def select_relevant_sentences(self, verbalized_claims_df, sentences_df, top_k=5):
        """
        Select most relevant sentences for each verbalized claim
        
        Args:
            verbalized_claims_df: DataFrame with 'verbalized_claim' column
            sentences_df: DataFrame with 'sentence' column
            top_k: Number of top sentences to return per claim
        
        Returns:
            DataFrame with evidence sentences and similarity scores
        """
        if verbalized_claims_df.empty or sentences_df.empty:
            return pd.DataFrame()
        
        evidence_data = []
        
        # Get all sentences
        sentences = sentences_df['sentence'].tolist()
        if not sentences:
            return pd.DataFrame()
        
        # Encode all sentences once for efficiency
        try:
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error encoding sentences: {e}")
            return pd.DataFrame()
        
        for _, claim_row in verbalized_claims_df.iterrows():
            # Use verbalized claim text
            claim_text = claim_row.get('verbalized_claim', '')
            if not claim_text:
                continue
            
            try:
                # Encode the claim
                claim_embedding = self.model.encode([claim_text], convert_to_tensor=True)
                
                # Compute similarity
                from sentence_transformers import util
                similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
                
                # Get top-k sentences
                import numpy as np
                top_indices = np.argsort(similarities.cpu())[-top_k:][::-1]
                
                # Create evidence entries
                for i in top_indices:
                    similarity_score = float(similarities[i])
                    evidence_data.append({
                        'claim_id': claim_row.get('claim_id', ''),
                        'verbalized_claim': claim_text,
                        'sentence': sentences[i],
                        'similarity_score': similarity_score,
                        'entity_id': claim_row.get('entity_id', ''),
                        'property_id': claim_row.get('property_id', ''),
                        'object_id': claim_row.get('object_id', ''),
                        'entity_label': claim_row.get('entity_label', ''),
                        'property_label': claim_row.get('property_label', ''),
                        'object_label': claim_row.get('object_label', ''),
                        'url': sentences_df.iloc[i]['url'] if 'url' in sentences_df.columns else '',
                        'reference_id': sentences_df.iloc[i].get('reference_id', '')
                    })
                    
            except Exception as e:
                logger.error(f"Error processing claim {claim_row.get('claim_id', '')}: {e}")
                continue
        
        return pd.DataFrame(evidence_data) if evidence_data else pd.DataFrame()
    
    def select_evidence(self, claim, sentences, top_k=5):
        """Legacy method for single claim evidence selection"""
        if not sentences:
            return []
        
        try:
            # Encode claim and sentences
            claim_embedding = self.model.encode([claim], convert_to_tensor=True)
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Compute similarity
            from sentence_transformers import util
            similarities = util.cos_sim(claim_embedding, sentence_embeddings)[0]
            
            # Get top-k sentences
            import numpy as np
            top_indices = np.argsort(similarities.cpu())[-top_k:][::-1]
            evidence = [(sentences[i], float(similarities[i])) for i in top_indices]
            
            return evidence
        except Exception as e:
            logger.error(f"Error selecting evidence: {e}")
            return []

# Backward compatibility function
def process_evidence(sentences_df, parser_result):
    """Legacy function for backward compatibility"""
    selector = EvidenceSelector()
    verbalized_claims = parser_result['claims'].copy()
    verbalized_claims['verbalized_claim'] = verbalized_claims.apply(
        lambda row: f"{row['entity_label']} {row['property_label']} {row['object_label']}", 
        axis=1
    )
    return selector.select_relevant_sentences(verbalized_claims, sentences_df)