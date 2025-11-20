"""
Simple Wikidata client with proper User-Agent
"""
import requests
import pandas as pd
import json

class SimpleWikidataClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProVe/1.0 (https://example.com; contact@example.com) Wikidata Research Tool',
            'Accept': 'application/json'
        })
        self.base_url = "https://www.wikidata.org/wiki/Special:EntityData"
    
    def get_entity(self, qid):
        """Get entity data from Wikidata"""
        url = f"{self.base_url}/{qid}.json"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HTTP {response.status_code} for {qid}")
        except Exception as e:
            print(f"Error fetching {qid}: {e}")
        return None
    
    def extract_claims_and_urls(self, qid):
        """Extract claims and reference URLs from entity"""
        entity_data = self.get_entity(qid)
        if not entity_data or 'entities' not in entity_data or qid not in entity_data['entities']:
            return pd.DataFrame(), []
        
        entity = entity_data['entities'][qid]
        claims_data = []
        reference_urls = []
        
        claims = entity.get('claims', {})
        for property_id, claim_list in claims.items():
            for claim in claim_list:
                # Extract claim information
                mainsnak = claim.get('mainsnak', {})
                datavalue = mainsnak.get('datavalue', {})
                
                claim_info = {
                    'qid': qid,
                    'property': property_id,
                    'value': json.dumps(datavalue.get('value', {})),
                    'claim_id': claim.get('id', '')
                }
                claims_data.append(claim_info)
                
                # Extract reference URLs (P854 = reference URL)
                for reference in claim.get('references', []):
                    for snak in reference.get('snaks', {}).get('P854', []):
                        url = snak.get('datavalue', {}).get('value', '')
                        if url and url.startswith('http'):
                            reference_urls.append(url)
        
        return pd.DataFrame(claims_data), list(set(reference_urls))
