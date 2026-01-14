import pandas as pd
import yaml
from tqdm import tqdm
from utils.logger import logger
from utils.textual_entailment_module import TextualEntailmentModule

class ClaimEntailmentChecker:
    def __init__(self, config_path: str = 'config.yaml', text_entailment=None):
        self.te_module = text_entailment or TextualEntailmentModule()

    def process_entailment(self, evidence_df: pd.DataFrame, html_df: pd.DataFrame, qid: str) -> pd.DataFrame:
        if evidence_df.empty: return pd.DataFrame()
        
        # Merge URLs
        if 'reference_id' not in evidence_df.columns: evidence_df['reference_id'] = evidence_df.index
        if 'reference_id' not in html_df.columns: html_df['reference_id'] = html_df.index
        
        # Join Evidence with HTML URLs
        evidence_df = evidence_df.merge(html_df[['reference_id', 'url']], on='reference_id', how='left')
        
        # --- SIMPLE ENTAILMENT (Row by Row) ---
        final_probs = []
        final_labels = []
        
        for _, row in tqdm(evidence_df.iterrows(), total=evidence_df.shape[0], desc="Entailment"):
            claim = row.get('verbalized_claim', row.get('triple', ''))
            
            # Get score
            probs_dict = self.te_module.get_batch_scores(claims=[claim], evidence=[row['sentence']])[0]
            label = self.te_module.get_label_from_scores(probs_dict)
            
            final_probs.append(probs_dict)
            final_labels.append(label)

        # Assign Results directly to DataFrame
        evidence_df['label_probabilities'] = final_probs
        evidence_df['result'] = final_labels
        evidence_df['result_sentence'] = evidence_df['sentence']
        
        # Column cleanup
        cols = ['qid', 'claim_id', 'entity_label', 'triple', 'verbalized_claim', 
                'property_id', 'object_id', 'result', 'result_sentence', 'url', 
                'similarity_score', 'entity_processing_time', 'label_probabilities']
        
        all_cols = list(set(cols + [c for c in evidence_df.columns if any(m in c for m in ['time', 'num'])]))
        
        return evidence_df[[c for c in all_cols if c in evidence_df.columns]].copy()