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

# --- SPEED OPTIMIZATION ---
LIMIT_CLAIMS = 15
LIMIT_URLS = 15

def initialize_models():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("⏳ Loading Models...")
    return TextualEntailmentModule(), SentenceRetrievalModule(), VerbModule()

def save_results_categorized(all_results_df: pd.DataFrame, output_file: str):
    """
    Saves results categorized by Verdict for Metric Calculation.
    Output: { "metadata": {...}, "results": { "SUPPORTS": [...], "REFUTES": [...] } }
    """
    if all_results_df.empty: return

    # Normalize group key
    if 'verbalized_claim' in all_results_df.columns:
        group_key = 'verbalized_claim'
    elif 'claim_id' in all_results_df.columns:
        group_key = 'claim_id'
    else:
        all_results_df['temp_claim_id'] = [f"c{i}" for i in range(len(all_results_df))]
        group_key = 'temp_claim_id'

    # Initialize Categories
    categorized_data = {
        "SUPPORTS": [],
        "REFUTES": [],
        "NOT_ENOUGH_INFO": []
    }

    # Iterate by Claim
    # We group by (QID + Claim ID) to ensure uniqueness
    grouped = all_results_df.groupby(['qid', group_key])

    for (qid, claim_identifier), claim_group in grouped:
        # 1. Determine Verdict
        # Since 'result' is the same for all evidence of a claim, take the first one
        verdict = claim_group['result'].iloc[0]
        
        # Normalize Verdict String (Handle variations like 'NEI')
        if verdict not in categorized_data:
            verdict = "NOT_ENOUGH_INFO"

        # 2. Get Metadata
        claim_text = claim_group['verbalized_claim'].iloc[0] if 'verbalized_claim' in claim_group else str(claim_identifier)
        entity_label = claim_group['entity_label'].iloc[0] if 'entity_label' in claim_group else "Unknown"

        # 3. Compile Evidence List
        evidence_list = []
        for _, row in claim_group.iterrows():
            evidence_list.append({
                "url": row.get('url', ''),
                "sentence": row.get('result_sentence', ''),
                "retrieval_score": float(row.get('similarity_score', 0.0)),
                "entailment_probs": row.get('label_probabilities', {})
            })

        # 4. Create Claim Object
        # Calculate max support confidence for metrics
        try:
            # Simple heuristic: Take average confidence of the predicted label across evidence
            probs = [e['entailment_probs'].get(verdict, 0.0) for e in evidence_list]
            avg_confidence = sum(probs) / len(probs) if probs else 0.0
        except:
            avg_confidence = 0.0

        claim_object = {
            "qid": qid,
            "entity_label": entity_label,
            "claim_id": str(claim_identifier),
            "verbalized_claim": claim_text,
            "aggregated_confidence": avg_confidence,
            "evidence": evidence_list
        }

        categorized_data[verdict].append(claim_object)

    # 5. Add Global Metadata for Baseline Metrics
    final_output = {
        "metadata": {
            "total_claims_processed": len(grouped),
            "distribution": {
                "SUPPORTS": len(categorized_data["SUPPORTS"]),
                "REFUTES": len(categorized_data["REFUTES"]),
                "NOT_ENOUGH_INFO": len(categorized_data["NOT_ENOUGH_INFO"])
            }
        },
        "results": categorized_data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"✅ Categorized Metric JSON Saved: {output_file}")

def process_entity(qid: str, models: tuple) -> tuple:
    text_entailment, sentence_retrieval, verb_module = models
    
    # 1. Parse
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    if parser_result['claims'].empty or parser_result['urls'].empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # --- SPEED OPTIMIZATION ---
    if len(parser_result['claims']) > LIMIT_CLAIMS:
        logger.info(f"✂️ Limiting claims to {LIMIT_CLAIMS}")
        parser_result['claims'] = parser_result['claims'].head(LIMIT_CLAIMS)
    if len(parser_result['urls']) > LIMIT_URLS:
        logger.info(f"✂️ Limiting URLs to {LIMIT_URLS}")
        parser_result['urls'] = parser_result['urls'].head(LIMIT_URLS)
        
    # 2. Verbalize
    try:
        verbalized_list = verb_module.verbalise_claims(parser_result['claims'])
    except:
        verbalized_list = []

    # Check if Neural Model failed or returned garbage
    if not verbalized_list or len(verbalized_list) != len(parser_result['claims']):
        logger.info("⚠️ Neural Verbalization skipped/failed. Using Template Fallback.")
        
        # --- FIX: BETTER TEMPLATE FALLBACK ---
        # Uses the newly fetched labels: "Barack Obama" + "medical condition" + "asthma"
        parser_result['claims']['verbalized_claim'] = parser_result['claims'].apply(
            lambda row: f"{row['entity_label']} has {row['property_label']} {row['object_label']}.", 
            axis=1
        )
    else:
        parser_result['claims']['verbalized_claim'] = verbalized_list
    # 3. Filter URLs
    forbidden = ('.jpg', '.png', '.pdf', '.xml')
    parser_result['urls'] = parser_result['urls'][
        ~parser_result['urls']['url'].str.lower().str.endswith(forbidden)
    ].copy()

    # 4. Fast Fetching
    fetcher = HTMLFetcher()
    html_df = fetcher.fetch_all_html(parser_result['urls'], parser_result)
    
    # 5. Process Sentences
    processor = HTMLSentenceProcessor()
    sentences_data = []
    for _, row in html_df.iterrows():
        if row['status'] == 200 and row['html']:
            text = processor.html_to_text(row["html"])
            text = text[:10000] # Limit massive pages
            for s in processor.text_to_sentences(text):
                if 50 < len(s.strip()) < 500:
                    sentences_data.append({"url": row["url"], "sentence": s.strip(), "reference_id": row.get("reference_id")})

    if not sentences_data: return html_df, pd.DataFrame(), {}
    
    # 6. Select & Entail
    evidence_df = EvidenceSelector(sentence_retrieval=sentence_retrieval).select_relevant_sentences(parser_result['claims'], pd.DataFrame(sentences_data))
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