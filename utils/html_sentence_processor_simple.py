"""
Simple Robust HTML Sentence Processor for ProVe
"""
from bs4 import BeautifulSoup
import nltk
import os
import re

class HTMLSentenceProcessor:
    def __init__(self):
        # Ensure NLTK data is available
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        nltk.data.path.append(nltk_data_dir)
        
        # Use basic sentence tokenization that always works
        self.use_nltk = False
        try:
            nltk.data.find('tokenizers/punkt')
            self.use_nltk = True
        except LookupError:
            print("⚠️ Using simple sentence splitting (NLTK punkt not available)")
    
    def html_to_text(self, html_content):
        """Extract clean text from HTML"""
        if not html_content:
            return ""
        
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
            print(f"Error processing HTML: {e}")
            return ""
    
    def text_to_sentences(self, text):
        """Split text into sentences - robust implementation"""
        if not text:
            return []
        
        try:
            if self.use_nltk:
                # Use NLTK if available
                sentences = nltk.tokenize.sent_tokenize(text)
            else:
                # Simple regex-based sentence splitting
                sentences = re.split(r'[.!?]+', text)
            
            # Filter and clean sentences
            cleaned_sentences = []
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:  # Only keep substantial sentences
                    cleaned_sentences.append(clean_sentence)
            
            return cleaned_sentences
        except Exception as e:
            print(f"Error in sentence splitting: {e}")
            # Ultimate fallback - split by periods
            return [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    def process_html(self, html_content):
        """Process HTML content into sentences"""
        text = self.html_to_text(html_content)
        sentences = self.text_to_sentences(text)
        return sentences
