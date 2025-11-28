import json
import csv
import statistics

# Configuration
INPUT_FILENAME = 'prove_diverse_batch_1764222584.json'
OUTPUT_CSV = 'prove_data_extracted.csv'

# Domain mapping
DOMAIN_MAPPING = {
    "Q2277": "History",
    "Q76": "Politics & Leaders", "Q9682": "Politics & Leaders", "Q312": "Politics & Leaders",
    "Q23": "Politics & Leaders", "Q142": "Politics & Leaders", "Q191": "Politics & Leaders",
    "Q352": "Politics & Leaders",
    "Q913": "Science & Discoveries",
    "Q22686": "Cultural Heritage & Arts", "Q729": "Cultural Heritage & Arts", "Q881": "Cultural Heritage & Arts",
    "Q90": "Geography & Places", "Q220": "Geography & Places", "Q30": "Geography & Places",
    "Q145": "Geography & Places", "Q183": "Geography & Places", "Q17": "Geography & Places",
    "Q84": "Geography & Places", "Q60": "Geography & Places", "Q148": "Geography & Places",
    "Q64": "Geography & Places", "Q1748": "Geography & Places",
    "Q413": "Organizations & Institutions", "Q95": "Organizations & Institutions", "Q94": "Organizations & Institutions",
    "Q5891": "Concepts & Ideas",
    "Q16": "Countries", "Q39": "Countries", "Q40": "Countries"
}

def get_category_count(comp_result, category):
    """Safely gets the count of items in a specific result category (SUPPORTS, etc.)"""
    if not isinstance(comp_result, dict):
        return 0
    
    cat_data = comp_result.get(category)
    
    # Sometimes the API returns "processing error" string instead of a dict
    if isinstance(cat_data, str): 
        return 0
        
    # Check if 'result' exists and is a dictionary (which contains the actual items)
    if isinstance(cat_data, dict) and 'result' in cat_data:
        return len(cat_data['result'])
    
    return 0

def main():
    print(f"Loading {INPUT_FILENAME}...")
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILENAME}. Make sure it is in the same folder as this script.")
        return

    results = data.get('results', {})
    
    extracted_rows = []
    
    # Statistics collectors
    stats_verified = []
    stats_supports = []
    stats_refutes = []
    stats_nei = []
    
    print(f"Processing {len(results)} items...")

    for qid, info in results.items():
        description = info.get('description', 'Unknown')
        domain = DOMAIN_MAPPING.get(qid, 'Other')
        success = info.get('success', False)
        comp_result = info.get('comp_result')
        
        # Handle failed or error items
        if not success or comp_result == 'processing error' or not isinstance(comp_result, dict):
            status = "FAILED"
            error_msg = info.get('error') or (comp_result if isinstance(comp_result, str) else "Unknown Error")
            
            row = {
                'QID': qid,
                'Description': description,
                'Domain': domain,
                'Status': status,
                'Total Verified Claims': 0,
                'Supports': 0,
                'Refutes': 0,
                'Not Enough Info': 0,
                'Reference Score': 0,
                'Error Message': error_msg
            }
        else:
            status = "SUCCESS"
            
            # Extract counts
            n_supports = get_category_count(comp_result, 'SUPPORTS')
            n_refutes = get_category_count(comp_result, 'REFUTES')
            n_nei = get_category_count(comp_result, 'NOT ENOUGH INFO')
            total_verified = n_supports + n_refutes + n_nei
            
            # Extract score
            ref_score = comp_result.get('Reference_score', 0)
            if not isinstance(ref_score, (int, float)):
                ref_score = 0
                
            row = {
                'QID': qid,
                'Description': description,
                'Domain': domain,
                'Status': status,
                'Total Verified Claims': total_verified,
                'Supports': n_supports,
                'Refutes': n_refutes,
                'Not Enough Info': n_nei,
                'Reference Score': round(ref_score, 4),
                'Error Message': ''
            }
            
            # Add to stats if successful
            stats_verified.append(total_verified)
            stats_supports.append(n_supports)
            stats_refutes.append(n_refutes)
            stats_nei.append(n_nei)
            
        extracted_rows.append(row)

    # Write to CSV
    print(f"Writing data to {OUTPUT_CSV}...")
    headers = ['QID', 'Description', 'Domain', 'Status', 'Total Verified Claims', 
               'Supports', 'Refutes', 'Not Enough Info', 'Reference Score', 'Error Message']
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(extracted_rows)

    # Print Summary to Console
    print("-" * 50)
    print("SUMMARY REPORT")
    print("-" * 50)
    
    if stats_verified:
        print(f"Total Successful Items: {len(stats_verified)}")
        print(f"Total Failed Items:     {len(results) - len(stats_verified)}")
        print("\nAVERAGES (Successful Items Only):")
        print(f"  Avg Verified Claims: {statistics.mean(stats_verified):.2f}")
        print(f"  Avg Supports:        {statistics.mean(stats_supports):.2f}")
        print(f"  Avg Refutes:         {statistics.mean(stats_refutes):.2f}")
        print(f"  Avg Not Enough Info: {statistics.mean(stats_nei):.2f}")
    else:
        print("No successful items found to analyze.")
        
    print("-" * 50)
    print(f"Done! Open '{OUTPUT_CSV}' to see the full data.")

if __name__ == "__main__":
    main()