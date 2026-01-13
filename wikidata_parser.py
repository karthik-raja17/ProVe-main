import ast
import json
import logging
import os
from typing import List, Dict, Any
import pandas as pd
import requests
import yaml
from utils.logger import logger

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

class EntityProcessor:
    def __init__(self):
        self.user_agent = 'ProVe-Research-Tool/1.1 (bot-contact: your-email@example.com)'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 403:
                logger.error("Wikidata 403 Forbidden")
                return self._empty_return()
            
            data = response.json()
            entity = data.get('entities', {}).get(qid, {})
            if not entity: return self._empty_return()

            claims_data = []
            claims_refs_data = []
            refs_data = []

            entity_id = entity.get('id', qid)
            entity_label = entity.get('labels', {}).get('en', {}).get('value', entity_id)

            for property_id, claims in entity.get('claims', {}).items():
                for claim in claims:
                    mainsnak = claim.get('mainsnak', {})
                    claim_id = claim.get('id')
                    
                    object_id = None
                    value_str = "no-value"
                    
                    # --- CRITICAL FIX: CLEAN VALUE EXTRACTION ---
                    if mainsnak.get('snaktype') == 'value':
                        datavalue = mainsnak.get('datavalue', {})
                        dv_type = datavalue.get('type')
                        dv_value = datavalue.get('value')
                        
                        if dv_type == 'wikibase-entityid':
                            # It's a Q-ID (e.g., Q123). Save it to object_id to fetch label later.
                            object_id = dv_value.get('id')
                            value_str = object_id 
                        elif dv_type == 'time':
                            # It's a Date. Extract just the timestamp string.
                            # Input: {'time': '+1961-08-04T00:00:00Z', ...} -> Output: "1961-08-04"
                            raw_time = dv_value.get('time', '')
                            value_str = raw_time.replace('+', '').split('T')[0]
                        elif dv_type == 'quantity':
                            # It's a Number. Extract just the amount.
                            # Input: {'amount': '+123', ...} -> Output: "123"
                            value_str = dv_value.get('amount', '').replace('+', '')
                        elif dv_type == 'monolingualtext':
                            value_str = dv_value.get('text', '')
                        else:
                            # Fallback for strings
                            value_str = str(dv_value)
                    
                    # Store Data 
                    # Note: We temporarily set object_label = value_str. 
                    # If it's a QID, we will overwrite this with the real name in the WikidataParser class.
                    claims_data.append((
                        entity_id, entity_label, claim_id, claim.get('rank'),
                        property_id, mainsnak.get('datatype'), value_str, object_id, value_str
                    ))

                    # 3. References (Standard logic)
                    if 'references' in claim:
                        for ref in claim['references']:
                            ref_hash = ref.get('hash')
                            claims_refs_data.append((claim_id, ref_hash))
                            for ref_prop_id, snaks in ref.get('snaks', {}).items():
                                for i, snak in enumerate(snaks):
                                    # Clean reference values too
                                    if snak.get('snaktype') == 'value':
                                        ref_dv = snak.get('datavalue', {})
                                        if ref_dv.get('type') == 'wikibase-entityid':
                                            ref_val = ref_dv.get('value', {}).get('id')
                                        else:
                                            # Simple string conversion for refs is usually fine
                                            ref_val = str(ref_dv.get('value'))
                                    else:
                                        ref_val = snak.get('snaktype')
                                    refs_data.append((ref_hash, ref_prop_id, str(i), snak.get('datatype'), ref_val))

            return {
                'claims': pd.DataFrame(claims_data, columns=['entity_id', 'entity_label', 'claim_id', 'rank', 'property_id', 'datatype', 'datavalue', 'object_id', 'object_label']),
                'claims_refs': pd.DataFrame(claims_refs_data, columns=['claim_id', 'reference_id']),
                'refs': pd.DataFrame(refs_data, columns=['reference_id', 'reference_property_id', 'reference_index', 'reference_datatype', 'reference_value'])
            }

        except Exception as e:
            logger.error(f"Entity Process Error: {e}")
            return self._empty_return()

    def _empty_return(self):
        return {'claims': pd.DataFrame(), 'claims_refs': pd.DataFrame(), 'refs': pd.DataFrame()}

class PropertyFilter:
    def __init__(self):
        self.bad_datatypes = ['commonsMedia', 'external-id', 'globe-coordinate', 'url', 'geo-shape', 'math']

    def filter_properties(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        if claims_df.empty: return claims_df
        df = claims_df[claims_df['rank'] != 'deprecated'].copy()
        df = df[~df['datatype'].isin(self.bad_datatypes)]
        return df

class URLProcessor:
    def __init__(self):
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.headers = {'User-Agent': 'ProVe-Research-Tool/1.1'}

    def get_labels_batch(self, id_list: List[str]) -> Dict[str, str]:
        """
        Robustly fetches English labels using Wikidata's native Label Service.
        """
        if not id_list: return {}
        
        # Clean list: unique, strings only, starts with Q or P
        unique_ids = list(set([x for x in id_list if x and isinstance(x, str) and (x.startswith('Q') or x.startswith('P'))]))
        labels_map = {}
        
        chunk_size = 50
        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i:i+chunk_size]
            values_str = " ".join([f"wd:{uid}" for uid in chunk])
            
            # Use SERVICE wikibase:label for automatic language fallback
            query = f"""
            SELECT ?id ?idLabel WHERE {{
              VALUES ?id {{ {values_str} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,en-gb,en-ca,en-us,mul". }}
            }}
            """
            try:
                r = requests.get(self.sparql_endpoint, params={'format': 'json', 'query': query}, headers=self.headers, timeout=10)
                if r.status_code == 200:
                    for item in r.json()['results']['bindings']:
                        uid = item['id']['value'].split('/')[-1]
                        label = item.get('idLabel', {}).get('value')
                        # Only store if valid text and not just the ID itself
                        if label and label != uid:
                            labels_map[uid] = label
            except Exception: pass
        
        return labels_map

    def get_formatter_url(self, property_id: str) -> str:
        query = f"SELECT ?formatter_url WHERE {{ wd:{property_id} wdt:P1630 ?formatter_url. }}"
        try:
            r = requests.get(self.sparql_endpoint, params={'query': query, 'format': 'json'}, headers=self.headers, timeout=20)
            return r.json()['results']['bindings'][0]['formatter_url']['value']
        except: return 'no_formatter_url'

    @staticmethod
    def _reference_value_to_url(reference_value: str) -> str:
        if reference_value in ['novalue', 'somevalue']: return reference_value
        try:
            val = ast.literal_eval(reference_value)
            if isinstance(val, dict) and 'value' in val: return str(val['value'])
            return str(val)
        except: return str(reference_value)

    def process_urls(self, filtered_data: Dict) -> pd.DataFrame:
        try:
            claims_df = filtered_data['claims']
            claims_refs_df = filtered_data['claims_refs']
            refs_df = filtered_data['refs']
            
            if claims_df.empty or refs_df.empty: return pd.DataFrame()

            valid_claim_ids = claims_df['claim_id'].unique()
            valid_refs = claims_refs_df[claims_refs_df['claim_id'].isin(valid_claim_ids)]
            if valid_refs.empty: return pd.DataFrame()

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
                    ext_id_df['url'] = ext_id_df.apply(lambda x: x['formatter_url'].replace('$1', x['ext_id']), axis=1)
            
            combined_url_data = []
            if not url_df.empty: combined_url_data.append(url_df)
            if not ext_id_df.empty: combined_url_data.append(ext_id_df)
            
            if not combined_url_data: return pd.DataFrame()
            
            final_url_df = pd.concat(combined_url_data, ignore_index=True)
            final_url_df = final_url_df.drop_duplicates(subset=['reference_id', 'url'])

            return final_url_df[['reference_id', 'reference_property_id', 'url']].reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in process_urls: {e}")
            return pd.DataFrame()

class WikidataParser:
    def __init__(self):
        self.entity_processor = EntityProcessor()
        self.property_filter = PropertyFilter()
        self.url_processor = URLProcessor()
        self.processing_stats = {}

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        self.processing_stats = {'entity_id': qid}
        
        # 1. Fetch Raw Data
        data = self.entity_processor.process_entity(qid)
        claims = self.property_filter.filter_properties(data['claims'])
        
        if not claims.empty:
            # --- FIX: OVERWRITE RAW IDs WITH LABELS ---
            
            # Collect all IDs
            prop_ids = claims['property_id'].unique().tolist()
            obj_ids = claims[claims['object_id'].notna()]['object_id'].unique().tolist()
            all_ids = prop_ids + obj_ids
            
            # Fetch Labels from SPARQL
            label_map = self.url_processor.get_labels_batch(all_ids)
            
            # 1. Fix Property Labels (P106 -> "occupation")
            claims['property_label'] = claims['property_id'].apply(lambda x: label_map.get(x, x))
            
            # 2. Fix Object Labels (Q123 -> "screenwriter")
            # Logic: If it has an object_id (meaning it's a Q-item), look up the label.
            # If not (meaning it's a date or number), keep the cleaned value string.
            claims['object_label'] = claims.apply(
                lambda row: label_map.get(row['object_id'], row['object_label']) if row['object_id'] else row['object_label'],
                axis=1
            )
            
        urls = self.url_processor.process_urls(data)
        
        return {'claims': claims, 'urls': urls}