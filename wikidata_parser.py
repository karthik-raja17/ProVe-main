import ast
import json
import logging
import os
from typing import List, Dict, Any
import pandas as pd
import requests
import yaml

# Initialize logger
logger = logging.getLogger("prove")

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path): return {}
        with open(config_path, 'r') as file: return yaml.safe_load(file)

class LabelCache:
    """Robust JSON File Cache with Absolute Path handling"""
    def __init__(self, cache_file='labels_cache.json'):
        # Ensure we look in the current working directory as an absolute path
        self.cache_file = os.path.abspath(cache_file)
        self.cache = self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded {len(data)} labels from cache: {self.cache_file}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, key):
        return self.cache.get(str(key))
        
    def update(self, new_data):
        if new_data:
            self.cache.update(new_data)
            self.save()

class EntityProcessor:
    def __init__(self):
        self.user_agent = 'ProVe-Tool/2.1 (Research Tool)'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200: return self._empty_return()
            
            data = response.json()
            entity = data.get('entities', {}).get(qid, {})
            if not entity: return self._empty_return()

            claims_data, claims_refs_data, refs_data = [], [], []
            
            # Metadata for the root entity
            entity_id = entity.get('id', qid)
            entity_label = entity.get('labels', {}).get('en', {}).get('value', entity_id)

            for property_id, claims in entity.get('claims', {}).items():
                for claim in claims:
                    mainsnak = claim.get('mainsnak', {})
                    claim_id = claim.get('id')
                    object_id, value_str = None, "no-value"
                    
                    if mainsnak.get('snaktype') == 'value':
                        datavalue = mainsnak.get('datavalue', {})
                        dv_type = datavalue.get('type')
                        dv_value = datavalue.get('value')
                        
                        if dv_type == 'wikibase-entityid':
                            object_id = dv_value.get('id')
                            value_str = object_id 
                        elif dv_type == 'time':
                            value_str = dv_value.get('time', '').replace('+', '').split('T')[0]
                        elif dv_type == 'quantity':
                            value_str = dv_value.get('amount', '').replace('+', '')
                        elif dv_type == 'monolingualtext':
                            value_str = dv_value.get('text', '')
                        else:
                            value_str = str(dv_value)
                    
                    # Store data for the 'claims' DataFrame
                    claims_data.append((
                        entity_id, entity_label, claim_id, claim.get('rank'),
                        property_id, mainsnak.get('datatype'), value_str, object_id, value_str
                    ))

                    # Process References for evidence collection
                    if 'references' in claim:
                        for ref in claim['references']:
                            ref_hash = ref.get('hash')
                            claims_refs_data.append((claim_id, ref_hash))
                            for ref_prop_id, snaks in ref.get('snaks', {}).items():
                                for i, snak in enumerate(snaks):
                                    ref_val = str(snak.get('datavalue', {}).get('value')) if snak.get('snaktype') == 'value' else snak.get('snaktype')
                                    refs_data.append((ref_hash, ref_prop_id, str(i), snak.get('datatype'), ref_val))

            return {
                # Column name 'qid' used for consistency across ProVe pipeline
                'claims': pd.DataFrame(claims_data, columns=['qid', 'entity_label', 'claim_id', 'rank', 'property_id', 'datatype', 'datavalue', 'object_id', 'object_label']),
                'claims_refs': pd.DataFrame(claims_refs_data, columns=['claim_id', 'reference_id']),
                'refs': pd.DataFrame(refs_data, columns=['reference_id', 'reference_property_id', 'reference_index', 'reference_datatype', 'reference_value'])
            }
        except Exception as e: 
            logger.error(f"Error processing entity {qid}: {e}")
            return self._empty_return()

    def _empty_return(self):
        return {'claims': pd.DataFrame(), 'claims_refs': pd.DataFrame(), 'refs': pd.DataFrame()}

class PropertyFilter:
    def __init__(self):
        # Exclude non-textual or complex datatypes that don't verbalize well
        self.bad_datatypes = ['commonsMedia', 'external-id', 'globe-coordinate', 'url', 'geo-shape', 'math', 'tabular-data']
    
    def filter_properties(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        if claims_df.empty: return claims_df
        return claims_df[~claims_df['datatype'].isin(self.bad_datatypes)].copy()

class URLProcessor:
    def __init__(self):
        self.api_endpoint = "https://www.wikidata.org/w/api.php"
        self.headers = {'User-Agent': 'ProVe-Tool/2.1'}
        self.cache = LabelCache()

    def get_labels_via_api(self, id_list: List[str]) -> Dict[str, str]:
        unique_ids = list(set([x for x in id_list if x and isinstance(x, str) and (x.startswith('Q') or x.startswith('P'))]))
        if not unique_ids: return {}
        
        final_map = {}
        missing_ids = []

        # 1. Check local cache first
        for uid in unique_ids:
            cached = self.cache.get(uid)
            if cached:
                final_map[uid] = cached
            else:
                missing_ids.append(uid)
        
        # 2. Fetch missing IDs from Wikidata API
        if missing_ids:
            logger.info(f"🌍 Fetching {len(missing_ids)} labels from Wikidata API...")
            chunk_size = 50
            new_labels = {}
            for i in range(0, len(missing_ids), chunk_size):
                chunk = missing_ids[i:i+chunk_size]
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": "en",
                    "format": "json"
                }
                try:
                    r = requests.get(self.api_endpoint, params=params, headers=self.headers, timeout=15)
                    if r.status_code == 200:
                        entities = r.json().get('entities', {})
                        for eid, info in entities.items():
                            label = info.get('labels', {}).get('en', {}).get('value')
                            if label:
                                new_labels[eid] = label
                except Exception as e:
                    logger.error(f"API Label request failed: {e}")
            
            # Update cache with new results
            if new_labels:
                self.cache.update(new_labels)
                final_map.update(new_labels)
            
        return final_map

    def process_urls(self, filtered_data: Dict) -> pd.DataFrame:
        """Extracts valid URLs from references for evidence collection"""
        try:
            refs_df = filtered_data['refs']
            if refs_df.empty: return pd.DataFrame()
            
            # Filter for URL-based references
            url_df = refs_df[refs_df['reference_datatype'] == 'url'].copy()
            
            def clean_url(val):
                if isinstance(val, str) and val.startswith("{"):
                    try:
                        return ast.literal_eval(val).get('value', val)
                    except:
                        return val
                return val
                
            if not url_df.empty:
                url_df['url'] = url_df['reference_value'].apply(clean_url)
                return url_df[['reference_id', 'reference_property_id', 'url']].drop_duplicates()
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

class WikidataParser:
    def __init__(self):
        self.entity_processor = EntityProcessor()
        self.property_filter = PropertyFilter()
        self.url_processor = URLProcessor()

    def process_entity(self, qid: str) -> Dict[str, pd.DataFrame]:
        # Log processing status
        logger.info(f"--- Parsing Wikidata Entity: {qid} ---")
        
        # Fetch data
        raw_data = self.entity_processor.process_entity(qid)
        claims = self.property_filter.filter_properties(raw_data['claims'])
        
        if not claims.empty:
            # 1. Collect all IDs that need label resolution
            prop_ids = claims['property_id'].unique().tolist()
            obj_ids = claims[claims['object_id'].notna()]['object_id'].unique().tolist()
            all_ids = list(set(prop_ids + obj_ids))
            
            # 2. Batch resolve labels via Cache/API
            label_map = self.url_processor.get_labels_via_api(all_ids)
            
            # 3. Apply Labels to the dataframe
            claims['property_label'] = claims['property_id'].apply(lambda x: label_map.get(x, x))
            claims['object_label'] = claims.apply(
                lambda row: label_map.get(row['object_id'], row['object_label']) if row['object_id'] else row['object_label'],
                axis=1
            )
            
            # 4. Generate the 'triple' string used for verbalization
            # Format: Subject Label | Property Label | Object Label
            claims['triple'] = claims.apply(
                lambda row: f"{row['entity_label']} | {row['property_label']} | {row['object_label']}",
                axis=1
            )
            
        # Extract evidence URLs
        urls = self.url_processor.process_urls(raw_data)
        
        return {
            'claims': claims, 
            'urls': urls,
            'claims_refs': raw_data['claims_refs'],
            'refs': raw_data['refs']
        }