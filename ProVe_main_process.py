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
LIMIT_CLAIMS = 15
LIMIT_URLS = 15

def initialize_models():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("⏳ Loading Models...")
    return TextualEntailmentModule(), SentenceRetrievalModule(), VerbModule()

def save_results_categorized(all_results_df: pd.DataFrame, output_file: str):
    if all_results_df.empty: return

    group_key = 'claim_id'
    
    categorized_data = {"SUPPORTS": [], "REFUTES": [], "NOT_ENOUGH_INFO": []}
    grouped = all_results_df.groupby(['qid', group_key])

    for (qid, _), claim_group in grouped:
        verdict = claim_group['result'].iloc[0]
        if verdict not in categorized_data: verdict = "NOT_ENOUGH_INFO"

        claim_row = claim_group.iloc[0]
        claim_object = {
            "qid": str(qid),
            "entity_label": str(claim_row.get('entity_label', 'Unknown')),
            "claim_id": str(claim_row.get('claim_id', '')),
            "triple": str(claim_row.get('triple', '')),
            "verbalized_claim": str(claim_row.get('verbalized_claim', '')),
            "evidence": []
        }

        for _, row in claim_group.iterrows():
            claim_object['evidence'].append({
                "url": row.get('url', ''),
                "sentence": row.get('result_sentence', ''),
                "retrieval_score": float(row.get('similarity_score', 0.0)),
                "entailment_probs": row.get('label_probabilities', {})
            })

        categorized_data[verdict].append(claim_object)

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
        
    # 2. Verbalize
    triples = parser_result['claims']['triple'].tolist()
    logger.info(f"Verbalizing {len(triples)} claims...")
    try:
        parser_result['claims']['verbalized_claim'] = verb_module.verbalise_claims(triples)
    except:
        parser_result['claims']['verbalized_claim'] = verb_module.verbalise_claims(parser_result['claims'])

    # 3. Filter URLs
    forbidden = ('.jpg', '.png', '.pdf', '.xml')
    parser_result['urls'] = parser_result['urls'][~parser_result['urls']['url'].str.lower().str.endswith(forbidden)].copy()

    # 4. Fetch
    fetcher = HTMLFetcher()
    html_df = fetcher.fetch_all_html(parser_result['urls'], parser_result)
    
    # 5. Sentences
    processor = HTMLSentenceProcessor()
    sentences_data = []
    for _, row in html_df.iterrows():
        if row['status'] == 200 and row['html']:
            text = processor.html_to_text(row["html"])[:10000]
            for s in processor.text_to_sentences(text):
                if 50 < len(s.strip()) < 500:
                    sentences_data.append({"url": row["url"], "sentence": s.strip(), "reference_id": row.get("reference_id")})

    if not sentences_data: return html_df, pd.DataFrame(), {}
    
    # 6. Select Evidence
    evidence_df = EvidenceSelector(sentence_retrieval=sentence_retrieval).select_relevant_sentences(parser_result['claims'], pd.DataFrame(sentences_data))
    
    # --- CRITICAL FIX: Merge Metadata Back ---
    if not evidence_df.empty:
        # Create metadata dataframe
        claims_meta = parser_result['claims'][['claim_id', 'qid', 'entity_label', 'triple', 'verbalized_claim']].copy()
        
        # Force string types to ensure matching works
        evidence_df['claim_id'] = evidence_df['claim_id'].astype(str)
        claims_meta['claim_id'] = claims_meta['claim_id'].astype(str)
        
        # Merge
        evidence_df = evidence_df.merge(claims_meta, on='claim_id', how='left')
        
        # DEBUG: Verify columns
        if 'verbalized_claim' not in evidence_df.columns:
            logger.error(f"❌ MERGE FAILED. Columns: {evidence_df.columns}")
        else:
            logger.info("✅ Metadata merged successfully.")

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