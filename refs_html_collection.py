from typing import Dict, Any, List
import yaml
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from utils.logger import logger

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except:
        return {}

class HTMLFetcher:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.max_workers = 20  
        self.timeout = 5       
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def fetch_html_with_requests(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers, verify=False)
            if response.status_code == 200:
                return response.text
            return "" 
        except Exception:
            return ""

    def fetch_all_html(self, url_df: pd.DataFrame, parser_result: Dict) -> pd.DataFrame:
        result_df = url_df.copy()
        result_df['html'] = None
        result_df['status'] = 0

        urls = result_df['url'].tolist()
        indices = result_df.index.tolist()
        
        logger.info(f"🚀 Starting parallel fetch for {len(urls)} URLs...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.fetch_html_with_requests, url): idx 
                for url, idx in zip(urls, indices)
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    html_content = future.result()
                    if html_content:
                        result_df.at[idx, 'html'] = html_content
                        result_df.at[idx, 'status'] = 200
                    else:
                        result_df.at[idx, 'status'] = 500
                except Exception:
                    result_df.at[idx, 'status'] = 500

        return result_df