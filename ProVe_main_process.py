import os
import json
import pandas as pd
import logging
import time
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
    return TextualEntailmentModule(), SentenceRetrievalModule(), VerbModule()

def save_results_categorized(all_results_df: pd.DataFrame, output_file: str):
    if all_results_df.empty: return

    # 1. Clean and deduplicate columns
    all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]

    # --- FIX: Safety initialization for required columns ---
    if 'label_probabilities' not in all_results_df.columns:
        all_results_df['label_probabilities'] = [{} for _ in range(len(all_results_df))]
    
    # 2. CREATE READABLE IDS
    def make_readable(row):
        triple = str(row.get('triple', ''))
        if '|' in triple:
            parts = triple.split('|')
            return f"{parts[0].strip()} -> {parts[1].strip()} -> {parts[2].strip()}"
        return f"{row.get('entity_label', 'Unknown')} ({row.get('claim_id', 'ID')})"

    all_results_df['readable_id'] = all_results_df.apply(make_readable, axis=1)

    # 3. Aggregate by Claim
    unique_claims = []
    for readable_id, group in all_results_df.groupby('readable_id'):
        first_row = group.iloc[0]
        
        evidence_list = []
        for _, row in group.iterrows():
            evidence_list.append({
                "url": str(row.get('url', '')),
                "sentence": str(row.get('result_sentence', '')),
                "retrieval_score": float(row.get('similarity_score', 0.0)),
                "entailment_probs": row.get('label_probabilities', {})
            })

        unique_claims.append({
            'readable_name': readable_id,
            'qid': str(first_row.get('qid', 'N/A')),
            'property_id': str(first_row.get('property_id', 'N/A')),
            'object_id': str(first_row.get('object_id', 'N/A')),
            'verbalized_claim': str(first_row.get('verbalized_claim', '')),
            'result': str(first_row.get('result', 'NOT_ENOUGH_INFO')),
            'evidence': evidence_list,
            'processing_time': float(first_row.get('entity_processing_time', 0)),
            'max_retrieval_score': max([e['retrieval_score'] for e in evidence_list]) if evidence_list else 0,
            # This line caused the crash previously; now it's safe
            'label_probabilities': first_row.get('label_probabilities', {})
        })
    
    unique_claims_df = pd.DataFrame(unique_claims)

    # 4. Export to JSON
    final_output = {}
    for verdict in ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']:
        subset = unique_claims_df[unique_claims_df['result'] == verdict]
        if not subset.empty:
            final_output[verdict] = subset.set_index('readable_name').to_dict(orient='index')
        else:
            final_output[verdict] = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved Human-Readable Results: {output_file}")

def process_entity(qid: str, models: tuple) -> tuple:
    start_time = time.perf_counter()
    text_entailment, sentence_retrieval, verb_module = models
    
    # 1. Parse
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    if parser_result['claims'].empty: return pd.DataFrame(), pd.DataFrame(), {}

    if len(parser_result['claims']) > LIMIT_CLAIMS:
        parser_result['claims'] = parser_result['claims'].head(LIMIT_CLAIMS)

    # 2. Verbalize
    triples_input = [{"subject": str(r['entity_label']), "predicate": str(r['property_label']), "object": str(r['object_label'])} 
                     for _, r in parser_result['claims'].iterrows()]
    try:
        generated = verb_module.verbalise(triples_input)
        parser_result['claims']['verbalized_claim'] = generated
    except:
        parser_result['claims']['verbalized_claim'] = parser_result['claims']['triple']

    # 3. Fetch
    fetcher = HTMLFetcher()
    html_df = fetcher.fetch_all_html(parser_result['urls'].head(LIMIT_URLS), parser_result)
    
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
    
    # 5. Selection
    evidence_df = EvidenceSelector(sentence_retrieval=sentence_retrieval).select_relevant_sentences(parser_result['claims'], pd.DataFrame(sentences_data))
    
    # 6. Merge Metadata
    duration = time.perf_counter() - start_time
    if not evidence_df.empty:
        meta_cols = ['claim_id', 'qid', 'entity_label', 'triple', 'verbalized_claim', 'property_id', 'object_id']
        claims_meta = parser_result['claims'].copy()
        claims_meta['entity_processing_time'] = duration
        
        claims_meta['claim_id'] = claims_meta['claim_id'].astype(str)
        evidence_df['claim_id'] = evidence_df['claim_id'].astype(str)
        
        drop_cols = [c for c in meta_cols if c in evidence_df.columns and c != 'claim_id']
        evidence_df = evidence_df.drop(columns=drop_cols)
        evidence_df = evidence_df.merge(claims_meta, on='claim_id', how='left')

    # 7. Entailment
    entailment_results = ClaimEntailmentChecker(text_entailment=text_entailment).process_entailment(evidence_df, html_df, qid)
    return html_df, entailment_results, {}

if __name__ == "__main__":
    DIVERSE_WIKIDATA_INSTANCES = { #LEADERS 
        "Q76": "Barack Obama",
     "Q969": "Winston Churchill", "Q312": "Julius Caesar", "Q12892": "Queen Elizabeth II", "Q352963": "Mahatma Gandhi", # LANDMARKS 
     "Q84": "Eiffel Tower", "Q34374": "Colosseum", "Q924": "Taj Mahal", "Q11693": "Machu Picchu", "Q43361": "Statue of Liberty", # ARTISTS 
     "Q558": "Leonardo da Vinci", "Q338": "Vincent van Gogh", "Q543": "Frida Kahlo", "Q762": "Ludwig van Beethoven", "Q1339": "Mozart", # ANIMALS 
     "Q146": "House cat", "Q833": "Tiger", "Q7377": "Elephant", "Q11573": "Bald eagle", "Q470": "Blue whale" }
    OUTPUT_FILE = "prove_metrics_results.json"
    models = initialize_models() 
    all_results = []
    for qid, label in DIVERSE_WIKIDATA_INSTANCES.items():
        print(f"--- Processing {label} ---")
        try:
            _, results, _ = process_entity(qid, models)
            if results is not None and not results.empty: all_results.append(results)
        except Exception as e: print(f"❌ Error {qid}: {e}")
    
    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        save_results_categorized(full_df, OUTPUT_FILE)