import pandas as pd

from wikidata_parser import WikidataParser
from refs_html_collection import HTMLFetcher
from refs_html_to_evidences import HTMLSentenceProcessor, EvidenceSelector
from claim_entailment import ClaimEntailmentChecker
from utils.textual_entailment_module import TextualEntailmentModule
from utils.sentence_retrieval_module import SentenceRetrievalModule
from utils.verbalisation_module import VerbModule

def initialize_models():
    """Initialize all required models once"""
    text_entailment = TextualEntailmentModule()
    sentence_retrieval = SentenceRetrievalModule()
    verb_module = VerbModule()
    return text_entailment, sentence_retrieval, verb_module

def process_entity(qid: str, models: tuple) -> tuple:
    """
    Process a single entity with pre-loaded models
    """
    text_entailment, sentence_retrieval, verb_module = models
    
    # Get URLs and claims
    parser = WikidataParser()
    parser_result = parser.process_entity(qid)
    parser_stats = parser.get_processing_stats()
    
    # Check if URLs exist
    if 'urls' not in parser_result or parser_result['urls'].empty:
        # Return empty DataFrames and parser stats
        empty_df = pd.DataFrame()
        empty_results = pd.DataFrame()
        return empty_df, empty_results, parser_stats
    
    # Initialize processors - FIXED: Correct EvidenceSelector initialization
    selector = EvidenceSelector(
        sentence_retrieval=sentence_retrieval,  # Pass the actual model, not string
        verb_module=verb_module
    )
    checker = ClaimEntailmentChecker(text_entailment=text_entailment)
    
    # Fetch HTML content
    fetcher = HTMLFetcher(config_path='config.yaml')
    html_df = fetcher.fetch_all_html(parser_result['urls'], parser_result)
    
    # Check if there are any successful (status 200) URLs
    if not (html_df['status'] == 200).any():
        # Return current html_df with failed fetches, empty results and parser stats
        empty_results = pd.DataFrame()
        return html_df, empty_results, parser_stats

    # VERBALIZATION: Convert raw triples to natural language
    # Check if verbalize_claims method exists, otherwise use fallback
    if hasattr(verb_module, 'verbalize_claims'):
        verbalized_claims = verb_module.verbalize_claims(parser_result['claims'])
    else:
        # Fallback: create simple verbalized claims
        verbalized_claims = parser_result['claims'].copy()
        verbalized_claims['verbalized_claim'] = verbalized_claims.apply(
            lambda row: f"{row['entity_label']} {row['property_label']} {row['object_label']}", 
            axis=1
        )
    
    # Convert HTML to sentences
    processor = HTMLSentenceProcessor()
    sentences_data = []
    for idx, row in html_df.iterrows():
        if row['status'] == 200 and row['html'] is not None:
            # Convert HTML to text
            text_content = processor.html_to_text(row["html"])
            # Convert text to sentences
            sentences = processor.text_to_sentences(text_content)
            for sent in sentences:
                sentences_data.append({
                    "url": row["url"],
                    "sentence": sent,
                    "html": row["html"]
                })
    sentences_df = pd.DataFrame(sentences_data) if sentences_data else pd.DataFrame()
    
    # Process evidence selection with VERBALIZED claims
    evidence_df = selector.select_relevant_sentences(verbalized_claims, sentences_df)
    
    # Check entailment with metadata
    entailment_results = checker.process_entailment(evidence_df, html_df, qid)
    
    return html_df, entailment_results, parser_stats

# ... (all your existing imports and function definitions for initialize_models and process_entity)

if __name__ == "__main__":
    # Your 30 Diverse Instances
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

    # 1. Initialize models ONCE locally
    # No need to import from ProVe_main_process; just call the function defined above.
    models = initialize_models() 
    
    all_results = []

    print(f"Starting batch processing for {len(DIVERSE_WIKIDATA_INSTANCES)} entities...")

    # 2. Iterate through the dictionary
    for qid, label in DIVERSE_WIKIDATA_INSTANCES.items():
        print(f"Processing {label} ({qid})...")
        try:
            # 3. Call the function directly from this script
            html_df, entailment_results, parser_stats = process_entity(qid, models)
            
            if not entailment_results.empty:
                # Add metadata to track which entity the results belong to
                entailment_results['batch_entity_qid'] = qid
                entailment_results['batch_entity_label'] = label
                all_results.append(entailment_results)
                print(f"Successfully processed {len(entailment_results)} claims for {label}.")
            else:
                print(f"No claims with valid references found for {label}.")
                
        except Exception as e:
            print(f"Error processing {label}: {e}")

    # 4. Save combined results to a single local CSV
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv("batch_results.csv", index=False)
        print(f"Batch complete. Results saved to batch_results.csv")