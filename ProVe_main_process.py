import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Force it here
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
import pandas as pd
import logging
from wikidata_parser import WikidataParser
from refs_html_collection import HTMLFetcher
from refs_html_to_evidences import HTMLSentenceProcessor, EvidenceSelector
from claim_entailment import ClaimEntailmentChecker
from utils.textual_entailment_module import TextualEntailmentModule
from utils.sentence_retrieval_module import SentenceRetrievalModule
from utils.verbalisation_module import VerbModule

# Setup logging to match your console output
logger = logging.getLogger("prove")

def initialize_models():
    """Initialize all required models once on the reserved GPU"""
    text_entailment = TextualEntailmentModule()
    sentence_retrieval = SentenceRetrievalModule()
    verb_module = VerbModule()
    return text_entailment, sentence_retrieval, verb_module

def process_entity(qid: str, models: tuple) -> tuple:
    text_entailment, sentence_retrieval, verb_module = models
    
    # 1. Parsing
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    parser_stats = parser.get_processing_stats()
    
    if parser_result['claims'].empty or parser_result['urls'].empty:
        return pd.DataFrame(), pd.DataFrame(), parser_stats
    
    # 2. Verbalization (Returns a LIST of strings)
    logger.info(f"Verbalizing {len(parser_result['claims'])} claims...")
    verbalized_claims_list = verb_module.verbalise_claims(parser_result['claims'])
    
    # FIX: Use len() instead of .empty for the list
    if not verbalized_claims_list or len(verbalized_claims_list) == 0:
        logger.error(f"Verbalization failed for {qid}")
        return pd.DataFrame(), pd.DataFrame(), parser_stats

    # 3. HTML Fetching
    fetcher = HTMLFetcher(config_path='config.yaml')
    html_df = fetcher.fetch_all_html(parser_result['urls'], parser_result)
    
    success_count = len(html_df[html_df['status'] == 200])
    logger.info(f"HTML Fetching Result: {success_count} pages downloaded.")

    if html_df.empty or success_count == 0:
        return html_df, pd.DataFrame(), parser_stats

    # 4. Sentence Processing
    processor = HTMLSentenceProcessor()
    sentences_data = []
    for _, row in html_df.iterrows():
        if row['status'] == 200 and row['html']:
            try:
                text_content = processor.html_to_text(row["html"])
                sentences = processor.text_to_sentences(text_content)
                for sent in sentences:
                    sentences_data.append({"url": row["url"], "sentence": sent, "reference_id": row.get("reference_id")})
            except Exception as e:
                logger.error(f"Sentence error: {e}")

    if not sentences_data:
        return html_df, pd.DataFrame(), parser_stats
    
    sentences_df = pd.DataFrame(sentences_data)
    
    # 5. Evidence Selection
    selector = EvidenceSelector(sentence_retrieval=sentence_retrieval, verb_module=verb_module)
    
    # Convert list to Series so internal .empty calls work
    verbalized_series = pd.Series(verbalized_claims_list)
    
    evidence_df = selector.select_relevant_sentences(verbalized_series, sentences_df)
    logger.info(f"Evidence Selection: Found {len(evidence_df)} relevant candidates.")
    
    # 6. Entailment
    checker = ClaimEntailmentChecker(text_entailment=text_entailment)
    entailment_results = checker.process_entailment(evidence_df, html_df, qid)
    
    return html_df, entailment_results, parser_stats

if __name__ == "__main__":
    # Your reservation on circa01: GPU 2
    # Ensure indices are set to 0 in modules as discussed for Visibility masking
    
    DIVERSE_WIKIDATA_INSTANCES = {
        "Q2277": "Roman Empire", "Q76": "Barack Obama", "Q9682": "Volodymyr Zelenskyy",
        "Q312": "Angela Merkel", "Q23": "George Washington", "Q142": "Margaret Thatcher",
        "Q191": "Vladimir Lenin", "Q352": "Charles de Gaulle", "Q913": "Theory of Evolution",
        "Q22686": "Donald Trump", "Q729": "Wolfgang Amadeus Mozart", "Q881": "Victor Hugo",
        "Q90": "Paris", "Q220": "Rome", "Q30": "United States of America",
        "Q145": "United Kingdom", "Q183": "Germany", "Q17": "Japan",
        "Q84": "London", "Q60": "New York City", "Q148": "Beijing",
        "Q64": "Berlin", "Q1748": "Copenhagen", "Q413": "Microsoft",
        "Q95": "Google", "Q94": "Android", "Q5891": "Democracy",
        "Q16": "Canada", "Q39": "Switzerland", "Q40": "Austria"
    }

    # Output file configuration
    output_file = "batch_results.csv"

    # Load models once
    models = initialize_models() 
    all_results = []

    for qid, label in DIVERSE_WIKIDATA_INSTANCES.items():
        print(f"\n--- Processing {label} ({qid}) ---")
        try:
            html_df, entailment_results, parser_stats = process_entity(qid, models)
            
            if not entailment_results.empty:
                # Add metadata
                entailment_results['batch_entity_qid'] = qid
                entailment_results['batch_entity_label'] = label
                
                # --- LIVE SAVE LOGIC ---
                # Check if file exists to determine if we need to write the header
                file_exists = os.path.isfile(output_file)
                
                # Append to CSV (mode='a')
                entailment_results.to_csv(
                    output_file, 
                    mode='a', 
                    index=False, 
                    header=not file_exists
                )
                
                all_results.append(entailment_results)
                print(f"DONE: Found {len(entailment_results)} verifiable claims. Saved to {output_file}")
            else:
                print(f"SKIP: No verifiable evidence found for {label}.")
                
        except Exception as e:
            # This catches the 'NoneType' error or network errors without stopping the whole loop
            print(f"CRITICAL ERROR for {label}: {e}")
            logger.error(f"Stack trace for {label}:", exc_info=True)

    # Final Summary
    if all_results:
        print(f"\nBatch complete. All processed results are available in {output_file}")
    else:
        print("\nBatch finished with no results found.")