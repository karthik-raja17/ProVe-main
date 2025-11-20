"""
Patch for qwikidata to set proper User-Agent
"""
import requests
from qwikidata.linked_data_interface import get_entity_dict_from_api

# Patch the requests session to include User-Agent
original_get = requests.Session.get

def patched_get(self, url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = 'ProVe/1.0 (https://example.com; contact@example.com) Wikidata Research Tool'
    return original_get(self, url, **kwargs)

# Apply the patch
requests.Session.get = patched_get

def get_entity_with_ua(qid):
    """Get entity with proper User-Agent"""
    return get_entity_dict_from_api(qid)
