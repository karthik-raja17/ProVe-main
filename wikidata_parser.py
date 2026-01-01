import ast
import json
import logging
import os
from typing import List, Dict, Any

import nltk
import pandas as pd
import requests
import yaml
from qwikidata.linked_data_interface import get_entity_dict_from_api

# Initialize logger
from utils.logger import logger

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {'database': {'name': 'prove_database.db'}, 'parsing': {'reset_database': False}}
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @property
    def database_name(self) -> str:
        return self.config.get('database', {}).get('name')
    
    @property
    def reset_database(self) -> bool:
        return self.config.get('parsing', {}).get('reset_database', False)

class EntityProcessor:
    def __init__(self):
        # REQUIRED: Set a descriptive User-Agent to bypass Wikidata 403 Forbidden errors
        self.user_agent = 'ProVe-Research-Tool/1.1 (https://github.com/karthik-raja17/ProVe; karthikraja2021@gmail.com)'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        """Fetch entity data using Special:EntityData endpoint which is bot-friendly"""
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        
        # Globally update headers temporarily for library compatibility
        original_headers = requests.utils.default_headers().copy()
        try:
            requests.utils.default_headers().update({'User-Agent': self.user_agent})
            
            response = self.session.get(url, timeout=30)
            if response.status_code == 403:
                logger.error(f"403 Forbidden: Wikidata blocking bot. UA used: {self.user_agent}")
                return self._empty_return()
            response.raise_for_status()
            data = response.json()
            entity = data.get('entities', {}).get(qid, {})
        except Exception as e:
            logger.error(f"API Connection failed for {qid}: {str(e)}")
            return self._empty_return()
        finally:
            # Restore original headers
            requests.utils.default_headers().clear()
            requests.utils.default_headers().update(original_headers)
            
        if not entity:
            return self._empty_return()

        claims_data = []
        claims_refs_data = []
        refs_data = []

        entity_id = entity.get('id', qid)
        entity_label = entity.get('labels', {}).get('en', {}).get('value', f"No label ({entity_id})")

        for property_id, claims in entity.get('claims', {}).items():
            for claim in claims:
                mainsnak = claim.get('mainsnak', {})
                claim_id = claim.get('id')
                
                object_id = None
                value_str = "no-value"
                
                if mainsnak.get('snaktype') == 'value':
                    datavalue = mainsnak.get('datavalue', {})
                    dv_type = datavalue.get('type')
                    dv_value = datavalue.get('value')
                    
                    if dv_type == 'wikibase-entityid':
                        object_id = dv_value.get('id') if isinstance(dv_value, dict) else dv_value
                    value_str = str(datavalue)
                else:
                    value_str = mainsnak.get('snaktype', 'unknown')
                
                claims_data.append((
                    entity_id, entity_label, claim_id, claim.get('rank'),
                    property_id, mainsnak.get('datatype'), value_str, object_id
                ))

                if 'references' in claim:
                    for ref in claim['references']:
                        ref_hash = ref.get('hash')
                        claims_refs_data.append((claim_id, ref_hash))
                        for ref_prop_id, snaks in ref.get('snaks', {}).items():
                            for i, snak in enumerate(snaks):
                                ref_val = str(snak.get('datavalue')) if snak.get('snaktype') == 'value' else snak.get('snaktype')
                                refs_data.append((ref_hash, ref_prop_id, str(i), snak.get('datatype'), ref_val))

        return {
            'claims': pd.DataFrame(claims_data, columns=['entity_id', 'entity_label', 'claim_id', 'rank', 'property_id', 'datatype', 'datavalue', 'object_id']),
            'claims_refs': pd.DataFrame(claims_refs_data, columns=['claim_id', 'reference_id']),
            'refs': pd.DataFrame(refs_data, columns=['reference_id', 'reference_property_id', 'reference_index', 'reference_datatype', 'reference_value'])
        }

    def _empty_return(self):
        return {'claims': pd.DataFrame(), 'claims_refs': pd.DataFrame(), 'refs': pd.DataFrame()}

class PropertyFilter:
    def __init__(self):
        self.bad_datatypes = ['commonsMedia', 'external-id', 'globe-coordinate', 'url', 'wikibase-form', 'geo-shape', 'math', 'musical-notation', 'tabular-data', 'wikibase-sense']

    def filter_properties(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        if claims_df.empty: 
            return claims_df
            
        original_size = len(claims_df)
        
        # Keep the rank filter (Standard requirement)
        df = claims_df[claims_df['rank'] != 'deprecated'].copy()
        
        # Keep the basic datatype filter
        df = df[~df['datatype'].isin(self.bad_datatypes)]
        
        # --- TEMPORARY TEST: Comment out the property-id removal ---
        # properties_to_remove = self._load_properties_to_remove()
        # if properties_to_remove:
        #     df = df[~df['property_id'].isin(properties_to_remove)]
        
        # Keep the null-value filter
        df = df[~df['datavalue'].astype(str).isin(['somevalue', 'novalue', 'None'])]
        
        logger.info(f"Filtering Results: {original_size} -> {len(df)}")
        return df

    def _load_properties_to_remove(self) -> List[str]:
        try:
            with open('properties_to_remove.json', 'r') as f:
                data = json.load(f)
            return [item['id'] for item in data['general']]
        except: return []

class URLProcessor:
    def __init__(self):
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.headers = {'User-Agent': 'ProVe-Research-Tool/1.1 (karthikraja2021@gmail.com)'}

    def get_formatter_url(self, property_id: str) -> str:
        query = f"SELECT ?formatter_url WHERE {{ wd:{property_id} wdt:P1630 ?formatter_url. }}"
        try:
            r = requests.get(self.sparql_endpoint, params={'query': query, 'format': 'json'}, headers=self.headers, timeout=20)
            return r.json()['results']['bindings'][0]['formatter_url']['value']
        except: return 'no_formatter_url'

    @staticmethod
    def _reference_value_to_url(reference_value: str) -> str:
        """Helper function to clean up stringified JSON reference values."""
        if reference_value in ['novalue', 'somevalue']:
            return reference_value
        try:
            val = ast.literal_eval(reference_value)
            if isinstance(val, dict) and 'value' in val:
                return str(val['value'])
            return str(val)
        except:
            return str(reference_value)

    def process_urls(self, filtered_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        try:
            claims_df = filtered_data['claims']
            claims_refs_df = filtered_data['claims_refs']
            refs_df = filtered_data['refs']
            
            if claims_df.empty or refs_df.empty: 
                return pd.DataFrame()

            valid_claim_ids = claims_df['claim_id'].unique()
            valid_refs = claims_refs_df[claims_refs_df['claim_id'].isin(valid_claim_ids)]
            if valid_refs.empty: 
                return pd.DataFrame()

            valid_ref_ids = valid_refs['reference_id'].unique()
            refs_subset = refs_df[refs_df['reference_id'].isin(valid_ref_ids)].copy()
            
            # 1. Direct URLs
            url_df = refs_subset[refs_subset['reference_datatype'] == 'url'].copy()
            if not url_df.empty:
                url_df['url'] = url_df['reference_value'].apply(self._reference_value_to_url)
            
            # 2. External IDs
            ext_id_df = refs_subset[refs_subset['reference_datatype'] == 'external-id'].copy()
            if not ext_id_df.empty:
                ext_id_df['ext_id'] = ext_id_df['reference_value'].apply(self._reference_value_to_url)
                ext_id_df['formatter_url'] = ext_id_df['reference_property_id'].apply(self.get_formatter_url)
                ext_id_df = ext_id_df[ext_id_df['formatter_url'] != 'no_formatter_url'].copy()
                if not ext_id_df.empty:
                    ext_id_df['url'] = ext_id_df.apply(
                        lambda x: x['formatter_url'].replace('$1', x['ext_id']), axis=1
                    )
            
            combined_url_data = []
            if not url_df.empty: combined_url_data.append(url_df)
            if not ext_id_df.empty: combined_url_data.append(ext_id_df)
            
            if not combined_url_data: 
                return pd.DataFrame()
            
            final_url_df = pd.concat(combined_url_data, ignore_index=True)
            final_url_df = final_url_df.drop_duplicates(subset=['reference_id', 'url'])

            return final_url_df[['reference_id', 'reference_property_id', 'url']].reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in process_urls: {e}")
            return pd.DataFrame()

    def get_labels_from_sparql(self, entity_ids: List[str]) -> Dict[str, str]:
        if not entity_ids: return {}
        query = f"SELECT ?id ?label WHERE {{ wd:{entity_ids[0]} rdfs:label ?label . FILTER(LANG(?label) = 'en') BIND(wd:{entity_ids[0]} AS ?id) }}"
        try:
            r = requests.get(self.sparql_endpoint, params={'format': 'json', 'query': query}, headers=self.headers, timeout=20)
            return {res['id']['value'].split('/')[-1]: res['label']['value'] for res in r.json()['results']['bindings']}
        except: return {}

class WikidataParser:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.entity_processor = EntityProcessor()
        self.property_filter = PropertyFilter()
        self.url_processor = URLProcessor()
        self.processing_stats = {}

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        self.processing_stats = {'entity_id': qid, 'parsing_start_timestamp': pd.Timestamp.now()}
        
        entity_data = self.entity_processor.process_entity(qid)
        total_claims = len(entity_data['claims'])
        
        filtered_claims = self.property_filter.filter_properties(entity_data['claims'])
        
        if not filtered_claims.empty and filtered_claims['entity_label'].iloc[0].startswith('No label'):
            label_map = self.url_processor.get_labels_from_sparql([filtered_claims['entity_id'].iloc[0]])
            if qid in label_map: filtered_claims['entity_label'] = label_map[qid]

        result = {'claims': filtered_claims, 'claims_refs': entity_data['claims_refs'], 'refs': entity_data['refs']}
        
        url_data = self.url_processor.process_urls(result)
        result['urls'] = url_data
        
        self.processing_stats['total_claims'] = total_claims
        self.processing_stats['url_references'] = len(url_data)
        
        return result

    def get_processing_stats(self) -> Dict:
        return self.processing_stats

if __name__ == "__main__":
    parser = WikidataParser()
    res = parser.process_entity('Q76')
    print(f"Processed entity. Found {len(res['urls'])} reference URLs.")