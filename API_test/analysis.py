import requests
import json
import time
from typing import Dict, List, Optional

class ProVeBatchClient:
    def __init__(self, base_url: str = "https://kclwqt.sites.er.kcl.ac.uk"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def get_comp_result(self, qid: str) -> Dict:
        """Retrieve comprehensive results for a given item"""
        print(f"  Processing {qid}...")
        response = self.session.get(
            f"{self.base_url}/api/items/getCompResult", 
            params={"qid": qid},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def get_item_summary(self, qid: str) -> Dict:
        """Retrieve summary results for an item"""
        response = self.session.get(
            f"{self.base_url}/api/items/summary", 
            params={"qid": qid},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

DIVERSE_WIKIDATA_INSTANCES = {
    # History
    "Q2277": "Roman Empire",
    
    # Politics & Leaders
    "Q76": "Barack Obama",
    "Q9682": "Volodymyr Zelenskyy", 
    "Q312": "Angela Merkel",
    "Q23": "George Washington",
    "Q142": "Margaret Thatcher",
    "Q191": "Vladimir Lenin",
    "Q352": "Charles de Gaulle",
    
    # Science & Discoveries
    "Q913": "Theory of Evolution",
    
    # Cultural Heritage & Arts
    "Q22686": "Donald Trump",
    "Q729": "Wolfgang Amadeus Mozart",
    "Q881": "Victor Hugo",
    
    # Geography & Places
    "Q90": "Paris",
    "Q220": "Rome", 
    "Q30": "United States of America",
    "Q145": "United Kingdom",
    "Q183": "Germany",
    "Q17": "Japan",
    "Q84": "London",
    "Q60": "New York City",
    "Q148": "Beijing",
    "Q64": "Berlin",
    "Q1748": "Copenhagen",
    
    # Organizations & Institutions
    "Q413": "Microsoft",
    "Q95": "Google",
    "Q94": "Android",
    
    # Concepts & Ideas
    "Q5891": "Democracy",

    
    # Countries 
    "Q16": "Canada",
    "Q39": "Switzerland",
    "Q40": "Austria"  
    
    
}

def run_batch_experiment():
    """Run ProVe verification on all 30 diverse Wikidata instances"""
    print("ProVe Batch Experiment - 30 Diverse Wikidata Instances")
    print("=" * 60)
    
    client = ProVeBatchClient()
    
    all_results = {}
    successful_items = []
    failed_items = []
    
    total_items = len(DIVERSE_WIKIDATA_INSTANCES)
    current = 0
    
    for qid, description in DIVERSE_WIKIDATA_INSTANCES.items():
        current += 1
        print(f"\n[{current}/{total_items}] Testing: {qid} - {description}")
        print("-" * 50)
        
        result = {
            'qid': qid,
            'description': description,
            'success': False,
            'comp_result': None,
            'summary': None,
            'error': None
        }
        
        try:
            # Get comprehensive results
            comp_result = client.get_comp_result(qid)
            result['comp_result'] = comp_result
            result['success'] = True
            successful_items.append(qid)
            
            # Basic analysis
            total_claims = comp_result.get('total_claims', 'Unknown')
            reference_score = comp_result.get('Reference_score', 0)
            supports = comp_result.get('SUPPORTS', {})
            refutes = comp_result.get('REFUTES', {})
            
            support_count = len(supports.get('result', {})) if isinstance(supports, dict) else 0
            refute_count = len(refutes.get('result', {})) if isinstance(refutes, dict) else 0
            
            print(f"   SUCCESS")
            print(f"    Total claims: {total_claims}")
            print(f"    Reference score: {reference_score}")
            print(f"    Supporting: {support_count}, Refuting: {refute_count}")
            
            # Try to get summary
            try:
                summary = client.get_item_summary(qid)
                result['summary'] = summary
                print(f"    Summary: Available")
            except:
                print(f"    Summary: Failed")
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}"
            result['error'] = error_msg
            failed_items.append(qid)
            print(f"   FAILED: {error_msg}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            result['error'] = error_msg
            failed_items.append(qid)
            print(f"   FAILED: {error_msg}")
        
        all_results[qid] = result
        
        # Be gentle with the API - add delay between requests
        time.sleep(2)
    
    # Save comprehensive results
    timestamp = int(time.time())
    results_file = f"prove_diverse_batch_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            'metadata': {
                'total_items': total_items,
                'successful': len(successful_items),
                'failed': len(failed_items),
                'success_rate': len(successful_items) / total_items * 100,
                'timestamp': timestamp
            },
            'results': all_results
        }, f, indent=2, default=str)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print(" BATCH EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total items processed: {total_items}")
    print(f" Successful: {len(successful_items)}")
    print(f" Failed: {len(failed_items)}")
    print(f" Success rate: {len(successful_items)/total_items*100:.1f}%")
    print(f" Results saved: {results_file}")
    
    # Domain-wise breakdown
    domains = {
        'Politics & Leaders': ['Q76', 'Q9682', 'Q312'],
        'Science & Discoveries': ['Q913', 'Q7187', 'Q323', 'Q808', 'Q1043', 'Q3908'],
        'Cultural Heritage & Arts': ['Q12483', 'Q860', 'Q22686', 'Q729', 'Q1156', 'Q881', 'Q33986'],
        'Geography & Places': ['Q90', 'Q220', 'Q30', 'Q11344'],
        'Organizations & Institutions': ['Q8337', 'Q12582', 'Q79859', 'Q413'],
        'Historical Events': ['Q1744', 'Q342', 'Q35470'],
        'Concepts & Ideas': ['Q5891', 'Q7889', 'Q131003', 'Q1748'],
        'Additional': ['Q9368', 'Q1339']
    }
    
    print(f"\n  DOMAIN-WISE BREAKDOWN:")
    for domain, items in domains.items():
        domain_success = [qid for qid in items if qid in successful_items]
        domain_failed = [qid for qid in items if qid in failed_items]
        print(f"  {domain}: {len(domain_success)} {len(domain_failed)}")
    
    # Show some key metrics from successful items
    if successful_items:
        print(f"\n KEY METRICS FROM SUCCESSFUL ITEMS:")
        total_supports = 0
        total_refutes = 0
        scores = []
        
        for qid in successful_items:
            comp = all_results[qid]['comp_result']
            supports = comp.get('SUPPORTS', {})
            refutes = comp.get('REFUTES', {})
            
            total_supports += len(supports.get('result', {})) if isinstance(supports, dict) else 0
            total_refutes += len(refutes.get('result', {})) if isinstance(refutes, dict) else 0
            
            score = comp.get('Reference_score', 0)
            if isinstance(score, (int, float)):
                scores.append(score)
        
        print(f"  Total supporting references: {total_supports}")
        print(f"  Total refuting references: {total_refutes}")
        if scores:
            print(f"  Average reference score: {sum(scores)/len(scores):.3f}")
        
        # Show top 5 most verified items
        item_verification_counts = []
        for qid in successful_items:
            comp = all_results[qid]['comp_result']
            supports = comp.get('SUPPORTS', {})
            refutes = comp.get('REFUTES', {})
            nei = comp.get('NOT ENOUGH INFO', {})
            
            support_count = len(supports.get('result', {})) if isinstance(supports, dict) else 0
            refute_count = len(refutes.get('result', {})) if isinstance(refutes, dict) else 0
            nei_count = len(nei.get('result', {})) if isinstance(nei, dict) else 0
            
            total_verified = support_count + refute_count + nei_count
            item_verification_counts.append((qid, total_verified))
        
        item_verification_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"\n TOP 5 MOST VERIFIED ITEMS:")
        for qid, count in item_verification_counts[:5]:
            description = DIVERSE_WIKIDATA_INSTANCES[qid]
            print(f"  {qid} - {description}: {count} references")

if __name__ == "__main__":
    run_batch_experiment()
