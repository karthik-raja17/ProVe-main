"""
HTML Fetcher for ProVe
"""
import requests
import pandas as pd
import time
from urllib.parse import urlparse

class HTMLFetcher:
    def __init__(self, delay=1, timeout=30):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_url(self, url):
        """Fetch HTML content from URL"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            return {
                'url': url,
                'html_content': response.text if response.status_code == 200 else '',
                'status': response.status_code,
                'domain': urlparse(url).netloc
            }
        except Exception as e:
            return {
                'url': url,
                'html_content': '',
                'status': 0,
                'domain': urlparse(url).netloc,
                'error': str(e)
            }
    
    def fetch_urls(self, urls):
        """Fetch multiple URLs with delay"""
        results = []
        
        for i, url in enumerate(urls):
            print(f"   Fetching {i+1}/{len(urls)}: {url[:80]}...")
            
            result = self.fetch_url(url)
            results.append(result)
            
            if i < len(urls) - 1:
                time.sleep(self.delay)
        
        return pd.DataFrame(results)
