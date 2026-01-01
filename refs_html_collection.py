from typing import Dict, Any, List
import yaml
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd

from utils.logger import logger


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class HTMLFetcher:
    HTTP_ERROR_MESSAGES = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        408: "Request Timeout",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout"
    }

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize HTMLFetcher with configuration and private MongoDB"""
        from pymongo import MongoClient
        
        # 1. Connect to your private port
        self.client = MongoClient('mongodb://localhost:27018/', serverSelectionTimeoutMS=2000)
        self.db = self.client['prove_cache']  # Database name
        self.collection = self.db['html_content']  # Collection name
        
        # 2. Verify connection
        try:
            self.client.server_info()
            logger.info("✅ Connected to private MongoDB on port 27018")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}. Caching will be disabled.")
        
        self.config = load_config(config_path)
        self.fetching_driver = self.config.get('html_fetching', {}).get('fetching_driver', 'requests')
        self.batch_size = self.config.get('html_fetching', {}).get('batch_size', 20)
        self.delay = self.config.get('html_fetching', {}).get('delay', 1.0)
        self.timeout = self.config.get('html_fetching', {}).get('timeout', 50)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_error_message(self, status_code: int) -> str:
        """Get descriptive error message for HTTP status code"""
        return self.HTTP_ERROR_MESSAGES.get(status_code, "Unknown Error")
    
    def fetch_html_with_requests(self, url: str) -> str:
        """Fetch HTML content using requests library with robust headers"""
        try:
            # Browser-like headers to prevent 403 Forbidden errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = requests.get(
                url,
                timeout=self.timeout,
                headers=headers,
                verify=False  # Bypass SSL issues on shared servers
            )
            
            if response.status_code == 200:
                return response.text
            else:
                return f"Error: Status code {response.status_code}"
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return f"Error: {str(e)}"

    def fetch_html_with_selenium(self, url: str) -> str:
        """Fetch HTML content using selenium"""
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            with webdriver.Chrome(options=chrome_options) as driver:
                driver.set_page_load_timeout(self.timeout)
                driver.get(url)
                time.sleep(1)  # Short delay to ensure page loads
                return driver.page_source
        except Exception as e:
            logger.error(f"Selenium error for {url}: {e}")
            return f"Error: {str(e)}"

    def fetch_all_html(self, url_df: pd.DataFrame, parser_result: Dict) -> pd.DataFrame:
        """Fetch HTML with MongoDB caching support and forced requests fetching logic"""
        result_df = url_df.copy()
        result_df['html'] = None
        result_df['status'] = None
        result_df['lang'] = None
        result_df['fetch_timestamp'] = None

        for i, (idx, row) in enumerate(result_df.iterrows()):
            url = row['url']
            
            # 1. Check MongoDB Cache first
            try:
                cached_data = self.collection.find_one({"url": url})
                if cached_data:
                    result_df.at[idx, 'html'] = cached_data['html']
                    result_df.at[idx, 'status'] = 200
                    result_df.at[idx, 'fetch_timestamp'] = cached_data.get('timestamp')
                    logger.info(f"📦 Loaded from cache: {url}")
                    continue
            except Exception as e:
                logger.warning(f"Cache lookup failed for {url}: {e}")

            # 2. Rate limiting
            if i > 0 and i % self.batch_size == 0:
                time.sleep(self.delay)

            # 3. Perform the fetch
            logger.info(f"🌐 Fetching: {url}")
            try:
                fetch_start_time = pd.Timestamp.now()
                
                # FORCE REQUESTS HERE - Calling your requests helper
                html = self.fetch_html_with_requests(url)
                
                if html and "Error:" not in html:
                    status = 200
                    # 4. Save to MongoDB Cache
                    self.collection.update_one(
                        {"url": url},
                        {"$set": {
                            "url": url,
                            "html": html,
                            "timestamp": fetch_start_time
                        }},
                        upsert=True
                    )
                else:
                    status = 500 # Mark as failure or specific error code if extracted
                    logger.error(f"Fetch failed for {url}: {html}")

                result_df.at[idx, 'html'] = html
                result_df.at[idx, 'status'] = status
                result_df.at[idx, 'fetch_timestamp'] = fetch_start_time

            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                result_df.at[idx, 'status'] = 500
                result_df.at[idx, 'html'] = f"Error: {str(e)}"

        return result_df

    def get_property_labels(self, property_ids: List[str]) -> Dict[str, str]:
        """Fetch labels for Wikidata properties"""
        endpoint_url = "https://query.wikidata.org/sparql"
        query = f"""
        SELECT ?id ?label WHERE {{
          VALUES ?id {{ wd:{' wd:'.join(property_ids)} }}
          ?id rdfs:label ?label .
          FILTER(LANG(?label) = "en" || LANG(?label) = "mul")
        }}
        """
        return self._execute_sparql_query(query)

    def get_entity_labels(self, entity_ids: List[str]) -> Dict[str, str]:
        """Fetch labels for Wikidata entities"""
        endpoint_url = "https://query.wikidata.org/sparql"
        query = f"""
        SELECT ?id ?label WHERE {{
          VALUES ?id {{ wd:{' wd:'.join(entity_ids)} }}
          ?id rdfs:label ?label .
          FILTER(LANG(?label) = "en" || LANG(?label) = "mul")
        }}
        """
        return self._execute_sparql_query(query)

    def _execute_sparql_query(self, query: str) -> Dict[str, str]:
        """Execute SPARQL query and return results as a dictionary"""
        endpoint_url = "https://query.wikidata.org/sparql"
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; MyBot/1.0; mailto:your@email.com)'
        }
        
        try:
            r = requests.get(endpoint_url, 
                            params={'format': 'json', 'query': query},
                            headers=headers)
            r.raise_for_status()
            results = r.json()
            
            labels = {}
            for result in results['results']['bindings']:
                label = result['label']['value']
                entity_id = result['id']['value'].split('/')[-1]
                labels[entity_id] = label
            
            return labels
            
        except Exception as e:
            logger.error(f"Error fetching labels: {e}")
            return {}

            
if __name__ == "__main__":
    qid = 'Q42'
    
    # Get URLs from WikidataParser
    from wikidata_parser import WikidataParser
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    url_references_df = parser_result['urls']
    
    # Fetch HTML content with metadata
    fetcher = HTMLFetcher(config_path='config.yaml')
    result_df = fetcher.fetch_all_html(url_references_df, parser_result)
    
    print(f"Successfully processed {len(result_df)} URLs")
