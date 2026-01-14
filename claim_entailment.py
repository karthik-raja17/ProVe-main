import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime
from utils.logger import logger

class ClaimEntailmentChecker:
    def __init__(self, config_path: str = 'config.yaml', text_entailment=None):
        self.te_module = text_entailment

    def check_entailment(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        df = evidence_df.copy()
        te_data = {'evidence_TE_prob': [], 'evidence_TE_labels': []}

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Entailment"):
            claim = row.get('verbalized_claim', row.get('triple', ''))
            probs = self.te_module.get_batch_scores(claims=[claim], evidence=[row['sentence']])
            label = self.te_module.get_label_from_scores(probs[0])
            
            te_data['evidence_TE_prob'].append(probs)
            te_data['evidence_TE_labels'].append(label)

        df['evidence_TE_prob'] = te_data['evidence_TE_prob']
        df['result'] = te_data['evidence_TE_labels']
        df['result_sentence'] = df['sentence']
        return df

    def process_entailment(self, evidence_df: pd.DataFrame, html_df: pd.DataFrame, qid: str) -> pd.DataFrame:
        if evidence_df.empty: return pd.DataFrame()
        
        # Merge URLs
        evidence_df = evidence_df.merge(html_df[['reference_id', 'url']], on='reference_id', how='left')
        
        # Run Check
        results = self.check_entailment(evidence_df)
        
        # Final Verdict Aggregation
        def get_final(group):
            if 'SUPPORTS' in group['result'].values:
                return group[group['result'] == 'SUPPORTS'].iloc[0]
            return group.iloc[0]

        final_results = results.groupby('claim_id').apply(get_final).reset_index(drop=True)
        
        # Attach probabilities back
        final_results['label_probabilities'] = final_results['evidence_TE_prob'].apply(lambda x: x[0])

        # --- FIX IS HERE: Added 'label_probabilities' to this list ---
        cols = ['qid', 'claim_id', 'entity_label', 'triple', 'verbalized_claim', 
                'property_id', 'object_id', 'result', 'result_sentence', 'url', 
                'similarity_score', 'entity_processing_time', 'label_probabilities']
        
        # Preserve metric columns dynamically as well
        all_cols = list(set(cols + [c for c in evidence_df.columns if 'time' in c or 'num' in c]))
        
        return final_results[[c for c in all_cols if c in final_results.columns]].copy()