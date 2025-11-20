"""
Fixed HTML to Evidence processor with correct NLTK path
"""
import nltk
import os
import re
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

class EvidenceSelector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
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
