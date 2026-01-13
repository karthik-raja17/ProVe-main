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
        try: SCORE_THRESHOLD = self.config['evidence_selection']['score_threshold']
        except: SCORE_THRESHOLD = 0.0

        textual_entailment_df = evidence_df.copy()
        
        te_columns = {
            'evidence_TE_prob': [], 'evidence_TE_prob_weighted': [],
            'evidence_TE_labels': [], 'claim_TE_prob_weighted_sum': [],
            'claim_TE_label_weighted_sum': [], 'processed_timestamp': []
        }

        for _, row in tqdm(textual_entailment_df.iterrows(), total=textual_entailment_df.shape[0], desc="Checking Entailment"):
            # PRIORITY: Verbalized > Triple > Raw Claim
            claim = ""
            if 'verbalized_claim' in row and pd.notna(row['verbalized_claim']) and str(row['verbalized_claim']).strip():
                claim = str(row['verbalized_claim'])
            elif 'triple' in row and pd.notna(row['triple']):
                claim = str(row['triple'])
            else:
                # Last resort fallback to prevent crash
                claim = row.get('claim', 'Unknown Claim')

            evidence = [{'sentence': row['sentence'], 'score': row['similarity_score']}]
            
            evidence_TE_prob = self.te_module.get_batch_scores(claims=[claim], evidence=[e['sentence'] for e in evidence])
            evidence_TE_labels = [self.te_module.get_label_from_scores(s) for s in evidence_TE_prob]
            
            evidence_TE_prob_weighted = []
            for probs, ev in zip(evidence_TE_prob, evidence):
                if ev['score'] > SCORE_THRESHOLD:
                    evidence_TE_prob_weighted.append({k: v * ev['score'] for k, v in probs.items()})
            if not evidence_TE_prob_weighted:
                evidence_TE_prob_weighted = [{'SUPPORTS': 0.0, 'REFUTES': 0.0, 'NOT_ENOUGH_INFO': 0.0}]
            
            claim_TE_prob_weighted_sum = {'SUPPORTS': 0.0, 'REFUTES': 0.0, 'NOT_ENOUGH_INFO': 0.0}
            for wp in evidence_TE_prob_weighted:
                for label, val in wp.items():
                    key = label.upper()
                    if key == 'NEI': key = 'NOT_ENOUGH_INFO'
                    claim_TE_prob_weighted_sum[key] = claim_TE_prob_weighted_sum.get(key, 0.0) + val
            
            claim_TE_label_weighted_sum = self.te_module.get_label_from_scores(claim_TE_prob_weighted_sum)
            
            te_columns['evidence_TE_prob'].append(evidence_TE_prob)
            te_columns['evidence_TE_prob_weighted'].append(evidence_TE_prob_weighted)
            te_columns['evidence_TE_labels'].append(evidence_TE_labels)
            te_columns['claim_TE_prob_weighted_sum'].append(claim_TE_prob_weighted_sum)
            te_columns['claim_TE_label_weighted_sum'].append(claim_TE_label_weighted_sum)
            te_columns['processed_timestamp'].append(datetime.now().isoformat())

        for col, values in te_columns.items(): textual_entailment_df[col] = values
        return textual_entailment_df

    def format_results(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        results = evidence_df.copy()
        all_result = pd.DataFrame()
        
        for idx, row in results.iterrows():
            if not row['evidence_TE_prob']: continue

            aBox = pd.DataFrame({
                'qid': [row.get('qid', '')],
                'claim_id': [row.get('claim_id', '')],
                'verbalized_claim': [row.get('verbalized_claim', '')],
                'triple': [row.get('triple', '')],
                'entity_label': [row.get('entity_label', '')],
                'url': [row.get('url', '')],
                'similarity_score': [row['similarity_score']],
                'reference_id': [row.get('reference_id', '')],
                'processed_timestamp': [row.get('processed_timestamp')],
                'Results': [pd.DataFrame({
                    'sentence': [row['sentence']],
                    'TextEntailment': [row['evidence_TE_labels'][0]],
                })]
            })
            all_result = pd.concat([all_result, aBox], axis=0)

        return all_result.reset_index(drop=True)

    def get_final_verdict(self, aggregated_result: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in aggregated_result.iterrows():
            temp = row.Results
            if 'SUPPORTS' in temp.TextEntailment.values:
                sent = temp[temp['TextEntailment']=='SUPPORTS']['sentence'].iloc[0]
                results.append({'result': 'SUPPORTS', 'result_sentence': sent})
            else:
                if temp.TextEntailment.empty: results.append({'result': 'NOT_ENOUGH_INFO', 'result_sentence': ''})
                else:
                    res = temp.TextEntailment.mode()[0]
                    sent = temp[temp['TextEntailment']==res]['sentence'].iloc[0]
                    results.append({'result': res, 'result_sentence': sent})
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
        final_verdict = self.get_final_verdict(aggregated_results)
        aggregated_results = pd.concat([aggregated_results, final_verdict], axis=1)
        
        cols_to_keep = [
            'qid', 'entity_label', 'claim_id', 'verbalized_claim', 'triple', 
            'url', 'similarity_score', 'processed_timestamp', 
            'result', 'result_sentence', 'reference_id'
        ]
        
        final_results = aggregated_results[[c for c in cols_to_keep if c in aggregated_results.columns]].copy()
        
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