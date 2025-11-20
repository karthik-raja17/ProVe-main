"""
Fixed HTML Sentence Processor for ProVe
"""
from bs4 import BeautifulSoup
import nltk
import os

class HTMLSentenceProcessor:
    def __init__(self):
        # Ensure NLTK data is available
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        nltk.data.path.append(nltk_data_dir)
        
        # Try to load punkt_tab, fallback to regular punkt
        try:
            nltk.data.find('tokenizers/punkt_tab')
            self.tokenizer = nltk.tokenize.punkt_tab
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
                self.tokenizer = nltk.tokenize.sent_tokenize
            except LookupError:
                print("⚠️ No NLTK tokenizer available, using simple split")
                self.tokenizer = None
    
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
        """Split text into sentences"""
        if not text:
            return []
        
        try:
            if self.tokenizer:
                if hasattr(self.tokenizer, 'tokenize'):
                    sentences = self.tokenizer.tokenize(text)
                else:
                    sentences = self.tokenizer(text)
            else:
                # Fallback: simple period-based split
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            return sentences
        except Exception as e:
            print(f"Error tokenizing sentences: {e}")
            # Fallback to simple split
            return [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    def process_html(self, html_content):
        """Process HTML content into sentences"""
        text = self.html_to_text(html_content)
        sentences = self.text_to_sentences(text)
        return sentences
