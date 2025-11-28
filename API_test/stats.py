import json
import statistics
import math
import sys

def calculate_stats(name, data_list):
    """Calculates statistics for a list of numbers."""
    if not data_list:
        return f"{name},0,0,0,0,0,0"
    
    # Filter out None values and non-numeric types just in case
    clean_data = [x for x in data_list if isinstance(x, (int, float))]
    
    if not clean_data:
        return f"{name},0,0,0,0,0,0"

    total_val = sum(clean_data)
    avg = statistics.mean(clean_data)
    median = statistics.median(clean_data)
    mn = min(clean_data)
    mx = max(clean_data)
    
    # Standard Deviation requires at least two data points
    if len(clean_data) > 1:
        std_dev = statistics.stdev(clean_data)
    else:
        std_dev = 0.0

    # Formatting to match the user's request style (approximate decimal places)
    # Reference Score uses 3 decimals, others use 1
    if name == "Reference Score":
        return f"{name},{total_val:.3f},{avg:.3f},{median:.1f},{mn:.2f},{mx:.2f},{std_dev:.2f}"
    else:
        return f"{name},{int(total_val)},{avg:.1f},{median:.1f},{mn},{mx},{std_dev:.1f}"

def main():
    input_file_path = 'prove_diverse_batch_1764222584.json'
    output_file_path = 'prove_statistics.csv'
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return

    # Lists to hold the raw data for calculation
    verified_claims_counts = []
    supporting_refs_counts = []
    refuting_refs_counts = []
    nei_counts = []
    reference_scores = []

    results = data.get('results', {})

    for qid, item in results.items():
        # Skip items that failed or have processing errors
        if not item.get('success'):
            continue
            
        comp_result = item.get('comp_result')
        
        # Check if comp_result is a valid dictionary (some might be error strings)
        if not isinstance(comp_result, dict):
            continue

        # 2. Extract Score
        ref_score = comp_result.get('Reference_score', 0.0)
        
        # Skip items where Reference_score is not a number (e.g. "processing error")
        # This ensures we only include truly successful items in the stats
        if not isinstance(ref_score, (int, float)):
            continue

        # 1. Extract Counts
        # We count the number of keys in the 'result' dictionary for each category
        
        # Supports
        supports_data = comp_result.get('SUPPORTS', {})
        supports_dict = supports_data.get('result', {}) if isinstance(supports_data, dict) else {}
        n_supports = len(supports_dict)
        
        # Refutes
        refutes_data = comp_result.get('REFUTES', {})
        refutes_dict = refutes_data.get('result', {}) if isinstance(refutes_data, dict) else {}
        n_refutes = len(refutes_dict)
        
        # Not Enough Info
        nei_data = comp_result.get('NOT ENOUGH INFO', {})
        nei_dict = nei_data.get('result', {}) if isinstance(nei_data, dict) else {}
        n_nei = len(nei_dict)
        
        # Verified Claims (Sum of the three categories above)
        # Note: This excludes 'error' category which are claims that crashed during processing
        n_verified = n_supports + n_refutes + n_nei

        # Append to lists
        supporting_refs_counts.append(n_supports)
        refuting_refs_counts.append(n_refutes)
        nei_counts.append(n_nei)
        verified_claims_counts.append(n_verified)
        reference_scores.append(ref_score)

    # Generate CSV Output Lines
    lines = []
    lines.append("Metric,Total,Average,Median,Min,Max,Std. Dev")
    lines.append(calculate_stats("Verified Claims", verified_claims_counts))
    lines.append(calculate_stats("Supporting Refs", supporting_refs_counts))
    lines.append(calculate_stats("Refuting Refs", refuting_refs_counts))
    lines.append(calculate_stats("Not Enough Info", nei_counts))
    lines.append(calculate_stats("Reference Score", reference_scores))
    
    # Print to console
    for line in lines:
        print(line)
        
    # Write to CSV file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\nSuccessfully saved statistics to {output_file_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    main()