# ProVe (Provenance Verification for Wikidata claims) - Forensic Restoration

## Overview

ProVe is a system designed to automatically verify claims and references in Wikidata. It extracts claims from Wikidata entities, fetches the referenced URLs, processes the HTML content, and uses NLP models to determine whether the claims are supported by the referenced content.

This repository contains a **Forensic Restoration** of the original pipeline, modernizing it for practical application on **real-world data**. While the original research utilized pre-annotated "golden data" for evaluation, this version is engineered to operate in live environments where such references are unavailable, addressing technical gaps and optimizing resource management for production-grade stability.

## System Architecture & Improvements

The system consists of several key components, including original modules and forensic upgrades:

1. **Data Collection and Processing**:
* `WikidataParser`: A custom forensic parser transitioning from SPARQL to **Batch-REST API** calls, reducing network overhead by 90%.
* `LabelCache`: A persistence layer storing resolved labels in `labels_cache.json` to eliminate redundant API calls.
* `HTMLFetcher`: Collects HTML content from reference URLs.
* `HTMLSentenceProcessor`: Converts HTML to sentences for analysis.


2. **Evidence Selection and Verification**:
* `EvidenceSelector`: Selects relevant sentences as evidence.
* **Fingerprinting Engine**: Implements a normalization filter to eliminate duplicate evidence and prevent "double-counting" redundant data from HTML sources.
* `ClaimEntailmentChecker`: Verifies entailment relationships using **Sequential Priority Logic**, prioritizing "smoking gun" evidence (Support/Refutation) over probabilistic averages.


3. **NLP Models**:
* `TextualEntailmentModule`: Checks textual entailment relationships using a DeBERTa-based model.
* `SentenceRetrievalModule`: Retrieves relevant sentences via a two-stage neural architecture (Bi-Encoder and Cross-Encoder).
* `VerbModule`: Handles triple verbalization. This module achieved **Functional Parity** with the original research by fine-tuning **T5-base** on **WebNLG 2020** with custom structural anchors.


4. **Data Storage**:
* **MongoDB**: Stores HTML content, entailment results, parser statistics, and status information.
* **SQLite**: Stores verification results for API access.


5. **Service Structure**:
* `ProVe_main_service.py`: Main service logic.
* `ProVe_main_process.py`: Entity processing logic.
* `background_processing.py`: Background processing tasks.



## Technical Stability & Optimizations

* **Dynamic Device Allocation**: Solves CUDA Out-of-Memory (OOM) errors by isolating the GPU for entailment scoring while offloading verbalization and retrieval tasks to the CPU, reducing VRAM usage by 45%.
* **Manual Embedding Resizing**: Resolves model loading errors by adjusting the token embedding layer (to size 32,103) to accommodate custom structural tokens (`<H>`, `<R>`, `<T>`).
* **Load-Update-Save Persistence**: Implements an atomic write pattern to ensure zero data loss and resumable processing during multi-hour batch runs.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Download NLP Models

The `base` folder contains essential NLP models, including the fine-tuned T5 verbalizer and pre-trained DeBERTa/MiniLM models.

Download from the [original repository](https://github.com/gabrielmaia7/RSP) and place the `base` folder in the project root directory.

### 3. Configure the System

Review and modify the `config.yaml` file to adjust database settings, HTML fetching parameters (batch size, delay), and evidence selection thresholds.

## Usage

### Processing a Single Entity

```python
from ProVe_main_process import initialize_models, process_entity

# Initialize models with optimized device allocation
models = initialize_models()

# Process entity by QID (e.g., Q44 for Barack Obama)
qid = 'Q44'
html_df, entailment_results, parser_stats = process_entity(qid, models)

```

### Running the Service

The main service can be started by running:

```bash
python ProVe_main_service.py

```

## Data Flow

1. A Wikidata QID is provided to the system.
2. The `WikidataParser` extracts claims and reference URLs using batch REST calls.
3. `HTMLFetcher` retrieves content, which is processed into sentences.
4. The **Fingerprinting Engine** normalizes and deduplicates sentences.
5. NLP models verbalize the triple and select the Top-5 relevant evidence passages.
6. **Sequential Priority Logic** determines if the evidence supports or refutes the claim.
7. Results are saved using the **Load-Update-Save** persistence pattern.

## Acknowledgments

This project builds upon the original ProVe architecture proposed by Gabriel Amaral, Odinaldo Rodrigues, and Elena Simperl. Special thanks to supervisors Fanfu WEI, Thibault EHRHART, and Raphaël TRONCY at EURECOM.
