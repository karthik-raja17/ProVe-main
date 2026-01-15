import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from utils.logger import logger
from utils.textual_entailment_module import TextualEntailmentModule

class ClaimEntailmentChecker:
    def __init__(self, config_path: str = 'config.yaml', text_entailment=None):
        self.te_module = text_entailment or TextualEntailmentModule()

    def process_entailment(self, evidence_df: pd.DataFrame, html_df: pd.DataFrame, qid: str) -> pd.DataFrame:
        if evidence_df.empty: return pd.DataFrame()
        
        # 1. Merge URLs
        if 'reference_id' not in evidence_df.columns: evidence_df['reference_id'] = evidence_df.index
        if 'reference_id' not in html_df.columns: html_df['reference_id'] = html_df.index
        
        evidence_df = evidence_df.merge(html_df[['reference_id', 'url']], on='reference_id', how='left')
        
        # 2. Calculate Entailment & Weighted Score (Aggregated by Claim)
        processed_rows = []
        
        # Group by claim to handle the "ensemble" of 5 sentences
        for claim_id, group in tqdm(evidence_df.groupby('claim_id'), desc="Entailment"):
            
            # Prepare Batch Inputs
            # Use 'verbalized_claim' if available, else 'triple'
            claim_text = group.iloc[0].get('verbalized_claim', group.iloc[0].get('triple', ''))
            sentences = group['sentence'].tolist()
            sim_scores = group['similarity_score'].tolist()
            
            # Run Model
            probs_list = self.te_module.get_batch_scores(claims=[claim_text]*len(sentences), evidence=sentences)
            
            # --- SCORE CALCULATION LOGIC ---
            # Initialize weighted accumulators
            w_support, w_refute, w_nei = 0.0, 0.0, 0.0
            total_weight = 0.0

            # Accumulate (Prob * Similarity)
            for i, probs in enumerate(probs_list):
                relevance = sim_scores[i]
                w_support += probs.get('SUPPORTS', 0.0) * relevance
                w_refute += probs.get('REFUTES', 0.0) * relevance
                w_nei += probs.get('NOT_ENOUGH_INFO', 0.0) * relevance
                total_weight += relevance
            
            # Normalize to get a valid probability distribution (Sum = 1.0)
            if total_weight > 0:
                final_support = w_support / total_weight
                final_refute = w_refute / total_weight
                final_nei = w_nei / total_weight
            else:
                final_support, final_refute, final_nei = 0.0, 0.0, 1.0

            # 1. Determine Verdict (Highest Score)
            avg_probs = {'SUPPORTS': final_support, 'REFUTES': final_refute, 'NOT_ENOUGH_INFO': final_nei}
            final_label = max(avg_probs, key=avg_probs.get)
            
            # 2. Calculate "Relevance Score" (-1 to 1)
            # Logic: Support(+1) - Refute(-1). NEI is neutral.
            relevance_score = final_support - final_refute

            # 3. Find Best Explanatory Sentence
            best_idx = 0
            max_contribution = -1
            for i, probs in enumerate(probs_list):
                contribution = probs.get(final_label, 0) * sim_scores[i]
                if contribution > max_contribution:
                    max_contribution = contribution
                    best_idx = i

            # 4. Create Result Rows
            for i, (idx, row) in enumerate(group.iterrows()):
                row_data = row.to_dict()
                
                row_data['evidence_probs'] = probs_list[i]  # Specific to this sentence
                row_data['label_probabilities'] = avg_probs # Aggregated consensus
                
                row_data['result'] = final_label
                row_data['relevance_score'] = relevance_score 
                row_data['result_sentence'] = sentences[best_idx]
                
                processed_rows.append(row_data)

        # 3. Finalize DataFrame
        result_df = pd.DataFrame(processed_rows)
        
        # --- CRITICAL FIX: ADDED 'sentence' TO PRESERVED COLUMNS ---
        cols = ['qid', 'claim_id', 'entity_label', 'triple', 'verbalized_claim', 
                'property_id', 'object_id', 'result', 'result_sentence', 'url', 
                'similarity_score', 'entity_processing_time', 
                'label_probabilities', 'evidence_probs', 'relevance_score', 'sentence']
        
        all_cols = list(set(cols + [c for c in result_df.columns if any(m in c for m in ['time', 'num'])]))
        
        return result_df[[c for c in all_cols if c in result_df.columns]].copy()