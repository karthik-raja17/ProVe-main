import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime
from utils.logger import logger
from utils.textual_entailment_module import TextualEntailmentModule

class ClaimEntailmentChecker:
    def __init__(self, config_path: str = 'config.yaml', text_entailment=None):
        self.logger = logger
        self.config = self.load_config(config_path)
        self.te_module = text_entailment or TextualEntailmentModule()
        
    @staticmethod
    def load_config(config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file: return yaml.safe_load(file)
        except: return {'evidence_selection': {'score_threshold': 0.0}}

    def check_entailment(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        try: SCORE = self.config['evidence_selection']['score_threshold']
        except: SCORE = 0.0

        df = evidence_df.copy()
        
        te_data = {'evidence_TE_prob': [], 'evidence_TE_prob_weighted': [], 'evidence_TE_labels': []}

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Entailment"):
            claim = ""
            if 'verbalized_claim' in row and pd.notna(row['verbalized_claim']) and str(row['verbalized_claim']).strip():
                claim = str(row['verbalized_claim'])
            elif 'triple' in row and pd.notna(row['triple']):
                claim = str(row['triple'])
            else:
                claim = "Unknown claim"

            evidence = row['sentence']
            score = row['similarity_score']
            
            probs_list = self.te_module.get_batch_scores(claims=[claim], evidence=[evidence])
            probs = probs_list[0]
            label = self.te_module.get_label_from_scores(probs)
            
            weighted = {k: v * score for k, v in probs.items()} if score > SCORE else {'SUPPORTS': 0.0, 'REFUTES': 0.0, 'NOT_ENOUGH_INFO': 0.0}
            
            te_data['evidence_TE_prob'].append([probs])
            te_data['evidence_TE_prob_weighted'].append([weighted])
            te_data['evidence_TE_labels'].append([label])

        for k, v in te_data.items(): df[k] = v
        return df

    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            if not isinstance(row['evidence_TE_prob'], list) or not row['evidence_TE_prob']: continue
            
            results.append({
                'qid': row.get('qid', ''),
                'claim_id': row.get('claim_id', ''),
                'property_id': row.get('property_id', 'N/A'),
                'object_id': row.get('object_id', 'N/A'),
                'verbalized_claim': row.get('verbalized_claim', ''),
                'triple': row.get('triple', ''),
                'entity_label': row.get('entity_label', ''),
                'url': row.get('url', ''),
                'similarity_score': row['similarity_score'],
                'label_probabilities': row['evidence_TE_prob'][0],
                # We calculate temporary result here
                'result': row['evidence_TE_labels'][0][0] if isinstance(row['evidence_TE_labels'][0], list) else row['evidence_TE_labels'][0],
                'result_sentence': row['sentence']
            })
        return pd.DataFrame(results)

    def get_final_verdict(self, aggregated_result: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in aggregated_result.iterrows():
            # Logic: If it supports, we take it. Otherwise we keep the existing label.
            if row['result'] == 'SUPPORTS':
                results.append({'result': 'SUPPORTS', 'result_sentence': row['result_sentence']})
            else:
                results.append({'result': row['result'], 'result_sentence': row['result_sentence']})
        return pd.DataFrame(results, index=aggregated_result.index)

    def process_entailment(self, evidence_df: pd.DataFrame, html_df: pd.DataFrame, qid: str) -> pd.DataFrame:
        if evidence_df.empty: return pd.DataFrame()
        
        if 'reference_id' not in evidence_df.columns: evidence_df['reference_id'] = evidence_df.index
        if 'reference_id' not in html_df.columns: html_df['reference_id'] = html_df.index
        evidence_df = evidence_df.merge(html_df[['reference_id', 'url']], on='reference_id', how='left')
        
        entailment_results = self.check_entailment(evidence_df)
        if entailment_results.empty: return pd.DataFrame()
        
        probabilities = entailment_results['evidence_TE_prob'].copy()
        
        aggregated_results = self.format_results(entailment_results)
        if aggregated_results.empty: return pd.DataFrame()

        final_verdict = self.get_final_verdict(aggregated_results)
        
        # --- FIX: Drop Duplicate Columns Before Merge ---
        # format_results has 'result' and 'result_sentence'. 
        # final_verdict ALSO has 'result' and 'result_sentence'.
        # We must drop them from aggregated_results to avoid duplicates (result_x, result_y) or collisions.
        cols_to_drop = [c for c in ['result', 'result_sentence'] if c in aggregated_results.columns]
        aggregated_results = aggregated_results.drop(columns=cols_to_drop)
        
        aggregated_results = pd.concat([aggregated_results, final_verdict], axis=1)
        
        cols_to_keep = [
            'qid', 'entity_label', 'claim_id', 'property_id', 'object_id', 
            'verbalized_claim', 'triple', 'url', 'similarity_score', 
            'processed_timestamp', 'result', 'result_sentence', 'reference_id'
        ]
        
        # Ensure cols exist
        for c in cols_to_keep:
            if c not in aggregated_results.columns: aggregated_results[c] = "N/A"

        final_results = aggregated_results[cols_to_keep].copy()
        
        def extract_prob(x):
            if not x or not isinstance(x, list) or not x[0]: 
                return {'SUPPORTS': 0.0, 'REFUTES': 0.0, 'NOT ENOUGH INFO': 0.0}
            return {
                'SUPPORTS': float(x[0].get('SUPPORTS', 0.0)),
                'REFUTES': float(x[0].get('REFUTES', 0.0)),
                'NOT ENOUGH INFO': float(x[0].get('NOT_ENOUGH_INFO', 0.0))
            }

        final_results['label_probabilities'] = probabilities.apply(extract_prob)
        return final_results