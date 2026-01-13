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
    return TextualEntailmentModule(), SentenceRetrievalModule(), VerbModule()

def save_results_categorized(all_results_df: pd.DataFrame, output_file: str):
    if all_results_df.empty: return

    # --- FIX: Deduplicate columns to prevent aggregation crashes ---
    all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]

    # Ensure required columns
    required_cols = ['qid', 'claim_id', 'property_id', 'object_id', 'triple', 'verbalized_claim', 'result', 'label_probabilities']
    for col in required_cols:
        if col not in all_results_df.columns:
            all_results_df[col] = "N/A"

    all_results_df['qid'] = all_results_df['qid'].astype(str)
    all_results_df['claim_id'] = all_results_df['claim_id'].astype(str)
    
    evidence_objects = []
    for _, row in all_results_df.iterrows():
        evidence_objects.append({
            "url": row.get('url', ''),
            "sentence": row.get('result_sentence', ''),
            "retrieval_score": float(row.get('similarity_score', 0.0)),
            "entailment_probs": row.get('label_probabilities', {})
        })
    all_results_df['evidence'] = evidence_objects

    # Robust Aggregation
    unique_claims = []
    for claim_id, group in all_results_df.groupby('claim_id'):
        first_row = group.iloc[0]
        evidence_list = group['evidence'].tolist()
        
        unique_claims.append({
            'claim_id': claim_id,
            'qid': first_row['qid'],
            'entity_label': first_row.get('entity_label', ''),
            'property_id': first_row['property_id'],
            'object_id': first_row['object_id'],
            'triple': first_row['triple'],
            'verbalized_claim': first_row['verbalized_claim'],
            'result': str(first_row['result']), # Ensure result is a string
            'evidence': evidence_list,
            'label_probabilities': first_row['label_probabilities']
        })
    
    unique_claims_df = pd.DataFrame(unique_claims)

    final_output = {}
    for verdict in ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']:
        subset = unique_claims_df[unique_claims_df['result'] == verdict]
        
        if not subset.empty:
            subset = subset.set_index('claim_id')
            final_output[verdict] = subset.to_dict(orient='dict')
        else:
            final_output[verdict] = {}

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
    logger.info(f"Verbalizing {len(parser_result['claims'])} claims...")
    
    triples_input = []
    for _, row in parser_result['claims'].iterrows():
        triples_input.append({
            "subject": str(row.get('entity_label', '')),
            "predicate": str(row.get('property_label', '')),
            "object": str(row.get('object_label', ''))
        })
    
    try:
        generated = verb_module.verbalise(triples_input)
        if isinstance(generated, str): generated = [generated]
        
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
    
    # 6. Merge Metadata
    if not evidence_df.empty:
        meta_cols = ['claim_id', 'qid', 'entity_label', 'triple', 'verbalized_claim', 'property_id', 'object_id']
        for col in meta_cols:
            if col not in parser_result['claims'].columns:
                parser_result['claims'][col] = "N/A"

        claims_meta = parser_result['claims'][meta_cols].copy()
        claims_meta['claim_id'] = claims_meta['claim_id'].astype(str)
        evidence_df['claim_id'] = evidence_df['claim_id'].astype(str)
        
        drop_cols = [c for c in meta_cols if c in evidence_df.columns and c != 'claim_id']
        evidence_df = evidence_df.drop(columns=drop_cols)
        
        evidence_df = evidence_df.merge(claims_meta, on='claim_id', how='left')
        evidence_df['verbalized_claim'] = evidence_df['verbalized_claim'].fillna(evidence_df['triple'])

    # 7. Entailment
    entailment_results = ClaimEntailmentChecker(text_entailment=text_entailment).process_entailment(evidence_df, html_df, qid)
    
    return html_df, entailment_results, {}

if __name__ == "__main__":
    DIVERSE_WIKIDATA_INSTANCES = {
    # LEADERS
    "Q76": "Barack Obama",
    "Q969": "Winston Churchill",
    "Q312": "Julius Caesar",
    "Q12892": "Queen Elizabeth II",
    "Q352963": "Mahatma Gandhi",
    # LANDMARKS
    "Q84": "Eiffel Tower",
    "Q34374": "Colosseum",
    "Q924": "Taj Mahal",
    "Q11693": "Machu Picchu",
    "Q43361": "Statue of Liberty",
    # ARTISTS
    "Q558": "Leonardo da Vinci",
    "Q338": "Vincent van Gogh",
    "Q543": "Frida Kahlo",
    "Q762": "Ludwig van Beethoven",
    "Q1339": "Mozart",
    # ANIMALS
    "Q146": "House cat",
    "Q833": "Tiger",
    "Q7377": "Elephant",
    "Q11573": "Bald eagle",
    "Q470": "Blue whale"
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