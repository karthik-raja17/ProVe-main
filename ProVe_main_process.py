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
    
    # --- CORRECTED URL FILTERING ---
    if not parser_result['urls'].empty:
        original_url_count = len(parser_result['urls'])
        
        # 1. Filter for specific domains
        allowed_domains = ['en.wikipedia.org', 'archive.org', 'web.archive.org', 'factcheck.org']
        
        # 2. Exclude binary/image files
        forbidden_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.mp4')

        # FIX: Added .str before .endswith()
        parser_result['urls'] = parser_result['urls'][
            parser_result['urls']['url'].str.contains('|'.join(allowed_domains), case=False, na=False) &
            ~parser_result['urls']['url'].str.lower().str.endswith(forbidden_extensions)
        ].copy()

        logger.info(f"🎯 URL Filter: Reduced {original_url_count} URLs to {len(parser_result['urls'])} high-quality sources.")
    # --- END OF FILTERING ---

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
                    # --- ADDED FILTERING HERE ---
                    # 1. Strip whitespace
                    clean_sent = sent.strip()
                    # 2. Only keep sentences between 20 and 500 characters
                    # This removes 'junk' (too short) and avoids CUDA 'out of bounds' (too long)
                    if 20 < len(clean_sent) < 300:
                        sentences_data.append({
                            "url": row["url"], 
                            "sentence": clean_sent, 
                            "reference_id": row.get("reference_id")
                        })
                    # --- END OF FILTERING ---
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
    # 1. Hardware Initialization
    # Ensure you have killed old processes: pkill -9 -u kandavel python
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    
    # 2. Testing Configuration
    DIVERSE_WIKIDATA_INSTANCES = {
        "Q76": "Barack Obama"
    }

    # 3. Output Configuration
    output_file = "batch_results.csv"
    # Optional: Remove old file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)

    # 4. Model Loading
    print("Initializing models (Retrieval on CPU, Entailment on GPU)...")
    models = initialize_models() 
    all_results = []

    # 5. Main Processing Loop
    for qid, label in DIVERSE_WIKIDATA_INSTANCES.items():
        print(f"\n--- Processing {label} ({qid}) ---")
        try:
            # Process the entity
            html_df, entailment_results, parser_stats = process_entity(qid, models)
            
            # 6. Result Handling & Live Saving
            if entailment_results is not None and not entailment_results.empty:
                print(f"📊 Attempting to save {len(entailment_results)} rows...")
                
                # Add metadata for analysis
                entailment_results['batch_entity_qid'] = qid
                entailment_results['batch_entity_label'] = label
                
                # Determine if we need to write the CSV header
                file_exists = os.path.isfile(output_file)
                
                # Perform the Live Save (Append mode)
                entailment_results.to_csv(
                    output_file, 
                    mode='a', 
                    index=False, 
                    header=not file_exists,
                    encoding='utf-8'
                )
                
                # Force the OS to write to disk immediately (prevents empty file on crash)
                if hasattr(os, 'sync'):
                    os.sync()
                
                all_results.append(entailment_results)
                print(f"✅ DONE: Found {len(entailment_results)} verifiable claims.")
                print(f"📁 Results written and synced to: {os.path.abspath(output_file)}")
            
            else:
                print(f"ℹ️ SKIP: No verifiable evidence found for {label}.")
                
        except Exception as e:
            # This prevents one entity's failure from stopping the whole batch
            print(f"❌ CRITICAL ERROR for {label}: {e}")
            logger.error(f"Stack trace for {label}:", exc_info=True)

    # 7. Final Summary
    if all_results:
        print(f"\n🚀 Batch complete. Final results available in {output_file}")
    else:
        print("\n⚠️ Batch finished, but no verifiable claims were found.")