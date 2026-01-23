import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from utils.logger import logger
from utils.textual_entailment_module import TextualEntailmentModule

class ClaimEntailmentChecker:
    def __init__(self, config_path: str = 'config.yaml', text_entailment=None):
        # Allow passing the existing loaded model to avoid reloading
        self.te_module = text_entailment or TextualEntailmentModule()

    def get_safe_verdict(self, probs_dict):
        """
        Applies Hard Gating to prevent Hallucinations.
        If the model isn't at least 85% confident in Support/Refute, it defaults to NEI.
        """
        # 1. Get the winner class and score
        winner = max(probs_dict, key=probs_dict.get)
        score = probs_dict[winner]
        
        # 2. Define Safety Threshold
        # 0.85 is a good balance. Lower it to 0.80 if you get too many NEIs.
        CONFIDENCE_THRESHOLD = 0.85 
        
        # 3. Apply Gate
        # If score is too low, or if the winner is purely based on weak signals, force NEI
        if score < CONFIDENCE_THRESHOLD:
            return "NOT_ENOUGH_INFO"
        
        return winner

    def process_entailment(self, evidence_df: pd.DataFrame, html_df: pd.DataFrame, qid: str) -> pd.DataFrame:
        if evidence_df.empty: return pd.DataFrame()
        
        # 1. Merge URLs for provenance tracking
        if 'reference_id' not in evidence_df.columns: evidence_df['reference_id'] = evidence_df.index
        if 'reference_id' not in html_df.columns: html_df['reference_id'] = html_df.index
        evidence_df = evidence_df.merge(html_df[['reference_id', 'url']], on='reference_id', how='left')
        
        processed_rows = []
        
        # Group by claim to handle the Top-K batch
        for claim_id, group in tqdm(evidence_df.groupby('claim_id'), desc="Entailment"):
            
            # Prepare Batch Inputs
            claim_text = group.iloc[0].get('verbalized_claim', group.iloc[0].get('triple', ''))
            sentences = group['sentence'].tolist()
            sim_scores = group['similarity_score'].tolist()
            
            # Run Batch Inference
            probs_list = self.te_module.get_batch_scores(claims=[claim_text]*len(sentences), evidence=sentences)
            
            # --- UPDATED SEQUENTIAL PRIORITY LOGIC ---
            found_support = False
            best_idx = 0
            final_label = "NOT_ENOUGH_INFO"
            
            # Step 1: Scan for ANY strong "SUPPORTS" (Priority 1)
            for i, probs in enumerate(probs_list):
                p_sup = probs.get('SUPPORTS', 0.0)
                if p_sup >= 0.85:  # High confidence threshold
                    final_label = "SUPPORTS"
                    best_idx = i
                    found_support = True
                    break  # Stop at the first supporting sentence
            
            # Step 2: Fallback to Weighted Jury if no support found (Priority 2)
            if not found_support:
                w_support, w_refute, w_nei, total_weight = 0.0, 0.0, 0.0, 0.0
                
                for i, probs in enumerate(probs_list):
                    relevance = sim_scores[i]
                    w_support += probs.get('SUPPORTS', 0.0) * relevance
                    w_refute += probs.get('REFUTES', 0.0) * relevance
                    w_nei += probs.get('NOT_ENOUGH_INFO', probs.get('NEI', 0.0)) * relevance
                    total_weight += relevance
                
                if total_weight > 0:
                    avg_probs = {
                        'SUPPORTS': w_support / total_weight,
                        'REFUTES': w_refute / total_weight,
                        'NOT_ENOUGH_INFO': w_nei / total_weight
                    }
                    final_label = self.get_safe_verdict(avg_probs)
                    
                    # Determine best index for the result sentence based on winner
                    best_idx = np.argmax([p.get(final_label, 0) for p in probs_list])
                else:
                    final_label = "NOT_ENOUGH_INFO"
                    best_idx = np.argmax(sim_scores)

            # Final Score Calculation
            final_probs = probs_list[best_idx]
            relevance_score = final_probs.get('SUPPORTS', 0.0) - final_probs.get('REFUTES', 0.0)

            # 4. Create Result Rows
            for i, (idx, row) in enumerate(group.iterrows()):
                row_data = row.to_dict()
                row_data['result'] = final_label
                row_data['result_sentence'] = sentences[best_idx]
                row_data['evidence_probs'] = probs_list[i]
                row_data['relevance_score'] = relevance_score 
                processed_rows.append(row_data)

        # 3. Finalize DataFrame with required columns
        result_df = pd.DataFrame(processed_rows)
        if result_df.empty: 
            return pd.DataFrame()

        # Define columns to preserve
        cols = ['qid', 'claim_id', 'entity_label', 'triple', 'verbalized_claim', 
                'result', 'result_sentence', 'url', 'similarity_score', 'sentence']
        
        # Filter columns and return
        return result_df[[c for c in cols if c in result_df.columns]].copy()