"""
Monkey patch for qwikidata to add proper User-Agent
"""
import requests
from qwikidata.linked_data_interface import get_entity_dict_from_api

# Store the original function
_original_get_entity_dict_from_api = get_entity_dict_from_api

def patched_get_entity_dict_from_api(
    entity_id, base_url='https://www.wikidata.org/wiki/Special:EntityData'
):
    """
    Patched version that includes proper User-Agent
    """
    # Store the original requests.get
    original_requests_get = requests.get
    
    def patched_requests_get(url, **kwargs):
        # Add User-Agent header if not already present
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = 'ProVe-Research-Tool/1.0 (https://github.com/your-username/ProVe; research@example.org)'
            kwargs['headers'] = headers
        return original_requests_get(url, **kwargs)
    
    # Temporarily replace requests.get with our patched version
    requests.get = patched_requests_get
    
    try:
        # Call the original function
        return _original_get_entity_dict_from_api(entity_id, base_url)
    finally:
        # Restore the original requests.get
        requests.get = original_requests_get

# Apply the patch
import qwikidata.linked_data_interface as ldi
ldi.get_entity_dict_from_api = patched_get_entity_dict_from_api

print("✅ Successfully patched qwikidata with proper User-Agent")
