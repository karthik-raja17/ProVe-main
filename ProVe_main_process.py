import os
import json
import pandas as pd
import logging
from wikidata_parser import WikidataParser
from refs_html_collection import HTMLFetcher
from refs_html_to_evidences import HTMLSentenceProcessor, EvidenceSelector
from claim_entailment import ClaimEntailmentChecker
from utils.textual_entailment_module import TextualEntailmentModule
from utils.sentence_retrieval_module import SentenceRetrievalModule
from utils.verbalisation_module import VerbModule

logger = logging.getLogger("prove")

# --- CONFIG ---
LIMIT_CLAIMS = 15
LIMIT_URLS = 15

def initialize_models():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("⏳ Loading Models...")
    # Initialize your custom module
    return TextualEntailmentModule(), SentenceRetrievalModule(), VerbModule()

def save_results_categorized(all_results_df: pd.DataFrame, output_file: str):
    if all_results_df.empty: return

    group_key = 'claim_id'
    categorized_data = {"SUPPORTS": [], "REFUTES": [], "NOT_ENOUGH_INFO": []}
    
    # Ensure grouping keys are strings
    all_results_df['qid'] = all_results_df['qid'].astype(str)
    all_results_df['claim_id'] = all_results_df['claim_id'].astype(str)
    
    grouped = all_results_df.groupby(['qid', group_key])

    for (qid, _), claim_group in grouped:
        verdict = claim_group['result'].iloc[0]
        if verdict not in categorized_data: verdict = "NOT_ENOUGH_INFO"

        row = claim_group.iloc[0]
        claim_obj = {
            "qid": str(qid),
            "entity_label": str(row.get('entity_label', 'Unknown')),
            "claim_id": str(row.get('claim_id', '')),
            "triple": str(row.get('triple', '')),
            "verbalized_claim": str(row.get('verbalized_claim', '')),
            "evidence": []
        }

        for _, r in claim_group.iterrows():
            claim_obj['evidence'].append({
                "url": r.get('url', ''),
                "sentence": r.get('result_sentence', ''),
                "retrieval_score": float(r.get('similarity_score', 0.0)),
                "entailment_probs": r.get('label_probabilities', {})
            })
        categorized_data[verdict].append(claim_obj)

    final_output = {"metadata": {"total": len(grouped)}, "results": categorized_data}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved: {output_file}")

def process_entity(qid: str, models: tuple) -> tuple:
    text_entailment, sentence_retrieval, verb_module = models
    
    # 1. Parse
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    
    if parser_result['claims'].empty or parser_result['urls'].empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    if len(parser_result['claims']) > LIMIT_CLAIMS:
        parser_result['claims'] = parser_result['claims'].head(LIMIT_CLAIMS)
    if len(parser_result['urls']) > LIMIT_URLS:
        parser_result['urls'] = parser_result['urls'].head(LIMIT_URLS)
        
    # 2. Verbalize (Using the 'verbalise' dict method)
    logger.info(f"Verbalizing {len(parser_result['claims'])} claims...")
    
    # Construct clean input for your module
    triples_input = []
    for _, row in parser_result['claims'].iterrows():
        triples_input.append({
            "subject": str(row.get('entity_label', '')),
            "predicate": str(row.get('property_label', '')),
            "object": str(row.get('object_label', ''))
        })
    
    try:
        # Call the robust method in your module
        generated = verb_module.verbalise(triples_input)
        
        # Handle single string return vs list return
        if isinstance(generated, str): generated = [generated]
        
        # Validate output
        if len(generated) != len(triples_input):
            logger.warning("Verbalizer count mismatch. Fallback to triples.")
            parser_result['claims']['verbalized_claim'] = parser_result['claims']['triple']
        else:
            parser_result['claims']['verbalized_claim'] = generated
            
    except Exception as e:
        logger.error(f"Verbalizer crashed: {e}. Fallback to triples.")
        parser_result['claims']['verbalized_claim'] = parser_result['claims']['triple']

    # 3. Fetch HTML
    forbidden = ('.jpg', '.png', '.pdf', '.xml')
    parser_result['urls'] = parser_result['urls'][~parser_result['urls']['url'].str.lower().str.endswith(forbidden)].copy()

    fetcher = HTMLFetcher()
    html_df = fetcher.fetch_all_html(parser_result['urls'], parser_result)
    
    # 4. Sentences
    processor = HTMLSentenceProcessor()
    sentences_data = []
    for _, row in html_df.iterrows():
        if row['status'] == 200 and row['html']:
            text = processor.html_to_text(row["html"])[:10000]
            for s in processor.text_to_sentences(text):
                if 50 < len(s.strip()) < 500:
                    sentences_data.append({"url": row["url"], "sentence": s.strip(), "reference_id": row.get("reference_id")})

    if not sentences_data: return html_df, pd.DataFrame(), {}
    
    # 5. Evidence Selection
    evidence_df = EvidenceSelector(sentence_retrieval=sentence_retrieval).select_relevant_sentences(parser_result['claims'], pd.DataFrame(sentences_data))
    
    # 6. Merge Metadata (Persist labels & verbalization)
    if not evidence_df.empty:
        meta_cols = ['claim_id', 'qid', 'entity_label', 'triple', 'verbalized_claim']
        claims_meta = parser_result['claims'][meta_cols].copy()
        
        # Type enforcement
        claims_meta['claim_id'] = claims_meta['claim_id'].astype(str)
        evidence_df['claim_id'] = evidence_df['claim_id'].astype(str)
        
        # Drop collisions
        drop_cols = [c for c in meta_cols if c in evidence_df.columns and c != 'claim_id']
        evidence_df = evidence_df.drop(columns=drop_cols)
        
        # Merge
        evidence_df = evidence_df.merge(claims_meta, on='claim_id', how='left')
        
        # Fill NaNs
        evidence_df['verbalized_claim'] = evidence_df['verbalized_claim'].fillna(evidence_df['triple'])

    # 7. Entailment
    entailment_results = ClaimEntailmentChecker(text_entailment=text_entailment).process_entailment(evidence_df, html_df, qid)
    
    return html_df, entailment_results, {}

if __name__ == "__main__":
    DIVERSE_WIKIDATA_INSTANCES = {
        "Q76": "Barack Obama"
    }
    OUTPUT_FILE = "prove_metrics_results.json"
    
    models = initialize_models() 
    all_results = []

    for qid, label in DIVERSE_WIKIDATA_INSTANCES.items():
        print(f"\n--- Processing {label} ({qid}) ---")
        try:
            _, results, _ = process_entity(qid, models)
            if results is not None and not results.empty:
                all_results.append(results)
                print(f"📊 Found {len(results)} items.")
        except Exception as e:
            print(f"❌ Error {qid}: {e}")

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        save_results_categorized(full_df, OUTPUT_FILE)
    else:
        print("⚠️ No results found.")