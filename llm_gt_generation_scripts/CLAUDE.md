# LLM Top-K Ground Truth

This project computes "ground-truth" LLM-as-a-judge rankings for BEIR benchmark datasets, enabling evaluation of document retrieval and ranking systems.

---

## Part I: IR-based Ranking

### Objective

The BEIR benchmark provides incomplete ground truth (only top-few relevant documents per query). This phase produces full rankings of all documents using an ensemble of state-of-the-art IR methods.

### Datasets

- **SciFact**: Scientific claim verification (small, ~5K docs)
- **SciDocs**: Scientific document similarity (small, ~25K docs)

### IR Ensemble

| Category | Model | Implementation |
|----------|-------|----------------|
| Sparse (classic) | BM25 | `rank_bm25` or Pyserini |
| Sparse (learned) | SPLADE++ | HuggingFace transformers |
| Dense | E5-large-v2 | sentence-transformers |
| Late interaction | ColBERTv2 | `colbert-ai` |
| Recent | BGE-large-en-v1.5 | sentence-transformers |

### Rank Aggregation

- **Primary**: Reciprocal Rank Fusion (RRF) with k=60
- **Secondary**: Borda count (for comparison)

### Stored Artifacts

- BEIR dataset name and metadata
- BEIR ground truth (qrels)
- Individual rankings from each ensemble member
- Aggregated rankings (RRF + Borda)

---

## Part II: Reranker-based Ranking

### Objective

Refine the top-k1 documents from IR aggregation using an ensemble of neural rerankers (cross-encoders). Cross-encoders jointly encode query-document pairs, enabling more nuanced relevance scoring than bi-encoder retrieval models.

### Reranker Ensemble

| Category | Model | Implementation |
|----------|-------|----------------|
| Cross-encoder (small) | ms-marco-MiniLM-L-6-v2 | sentence-transformers |
| Cross-encoder (medium) | ms-marco-MiniLM-L-12-v2 | sentence-transformers |
| Cross-encoder (large) | BGE-reranker-large | sentence-transformers |
| Cross-encoder (multilingual) | mMARCO-mMiniLM-L-12-H-384-v1 | sentence-transformers |

### Reranking Protocol

1. Select top-k1 documents (e.g., k1=1000) from Part I aggregated ranking
2. For each reranker:
   - Score all k1 query-document pairs
   - Produce per-reranker ranking
3. Aggregate reranker rankings via RRF or Borda
4. Output top-k1 refined ranking

### Rank Aggregation

- **Primary**: Reciprocal Rank Fusion (RRF) with k=60
- **Secondary**: Borda count (for comparison)

### Stored Artifacts

- Individual reranker scores and rankings for top-k1
- Aggregated reranker rankings (RRF + Borda)
- Metadata (models used, k1 value, aggregation params)

---

## Part III: LLM-based Refinement

### Objective

Refine top-k2 documents (k2 < k1, typically in the few 100s range) from reranker aggregation using pairwise LLM comparisons aggregated via ELO ratings.

### LLM Models

| Class | Model | Backend |
|-------|-------|---------|
| Small | Llama 3.1 8B | vLLM (local) |
| Small | Llama 3.2 11B | vLLM (local) |
| Oracle | GPT-4.5 | OpenAI API |

### Pairwise Comparison Protocol

1. Select top-k2 documents from Part II aggregated reranker ranking
2. Generate all unique pairs: k2*(k2-1)/2 comparisons
3. For each pair:
   - Randomize presentation order (A,B) vs (B,A)
   - Prompt includes: query, document metadata, both documents
   - Enable reasoning (chain-of-thought)
   - Constrained generation for final answer: `A` or `B`
4. Parse winner from response

### ELO Tournament

- Initialize ratings from reranker-based ranks
- Process all pairwise comparison results
- Run tournament iterations until convergence
- Store full ELO state history

### Embeddings

Extract and store document embeddings from LLM:
- Without query context
- With query context

### Stored Artifacts

- All pairwise comparison results (query, doc_a, doc_b, winner, reasoning, model)
- ELO state at each iteration
- Final LLM-refined rankings
- Document embeddings

---

## Pipeline Architecture

The pipeline consists of **6 independent phases**, each reading from cached outputs of previous phases and writing its own cached outputs. This enables:
- Independent execution of each phase
- Resumability at any phase boundary
- Easy re-running of later phases without re-running earlier ones
- Swapping components (e.g., different aggregation methods) without re-running retrieval

### Phase Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 1: IR Retrieval                                                      │
│  ─────────────────────                                                      │
│  Input:  data/raw/{dataset}/  (BEIR corpus, queries)                        │
│  Output: data/phase1_retrieval/{dataset}/{retriever}.parquet                │
│                                                                             │
│  Runs each retriever independently, caches per-retriever rankings.          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 2: IR Rank Aggregation                                               │
│  ────────────────────────────                                               │
│  Input:  data/phase1_retrieval/{dataset}/*.parquet                          │
│  Output: data/phase2_ir_aggregation/{dataset}/aggregated_{method}.parquet   │
│                                                                             │
│  Reads all retriever rankings, aggregates via RRF/Borda, caches results.    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 3: Reranker Scoring                                                  │
│  ─────────────────────────                                                  │
│  Input:  data/phase2_ir_aggregation/{dataset}/aggregated_{method}.parquet   │
│  Output: data/phase3_reranking/{dataset}/{reranker}.parquet                 │
│                                                                             │
│  Selects top-k1 from IR aggregation, scores with each reranker, caches      │
│  per-reranker rankings.                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 4: Reranker Rank Aggregation                                         │
│  ──────────────────────────────────                                         │
│  Input:  data/phase3_reranking/{dataset}/*.parquet                          │
│  Output: data/phase4_rerank_aggregation/{dataset}/aggregated_{method}.parquet│
│                                                                             │
│  Reads all reranker rankings, aggregates via RRF/Borda, caches results.     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 5: LLM Comparison Collection                                         │
│  ──────────────────────────────────                                         │
│  Input:  data/phase4_rerank_aggregation/{dataset}/aggregated_{method}.parquet│
│  Output: data/phase5_comparisons/{dataset}/{query_id}/comparisons.jsonl     │
│                                                                             │
│  Reads aggregated reranker rankings, selects top-k2, runs pairwise LLM      │
│  comparisons, appends each comparison result to JSONL (resumable).          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 6: ELO Tournament                                                    │
│  ───────────────────────                                                    │
│  Input:  data/phase5_comparisons/{dataset}/{query_id}/comparisons.jsonl     │
│  Output: data/phase6_rankings/{dataset}/{query_id}/final_ranking.parquet    │
│          data/phase6_rankings/{dataset}/{query_id}/elo_state.json           │
│          data/phase6_rankings/{dataset}/{query_id}/embeddings/              │
│                                                                             │
│  Reads all comparisons, runs ELO until convergence, stores final rankings   │
│  and embeddings.                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-topk-gt/
├── src/
│   ├── data/           # Data loading, models, storage
│   │   ├── models.py       # Pydantic data models
│   │   ├── beir_loader.py  # BEIR dataset loading
│   │   └── storage.py      # Read/write utilities for all phases
│   ├── retrieval/      # Phase 1: IR model implementations
│   │   ├── base.py         # Abstract Retriever interface
│   │   ├── bm25.py
│   │   ├── splade.py
│   │   ├── e5.py
│   │   ├── colbert.py
│   │   └── bge.py
│   ├── aggregation/    # Phases 2 & 4: Rank aggregation algorithms
│   │   ├── base.py         # Abstract Aggregator interface
│   │   ├── rrf.py          # Reciprocal Rank Fusion
│   │   └── borda.py        # Borda count
│   ├── reranking/      # Phase 3: Reranker model implementations
│   │   ├── base.py         # Abstract Reranker interface
│   │   ├── cross_encoder.py    # Generic cross-encoder wrapper
│   │   ├── minilm.py       # MiniLM-based rerankers
│   │   └── bge_reranker.py # BGE reranker
│   ├── llm/            # Phase 5: LLM backends & comparison logic
│   │   ├── base.py         # Abstract LLMBackend interface
│   │   ├── vllm_backend.py # vLLM for local models
│   │   ├── openai_backend.py
│   │   ├── pairwise.py     # Comparison prompt & parsing
│   │   └── embeddings.py   # Embedding extraction
│   ├── ranking/        # Phase 6: ELO system
│   │   └── elo.py
│   └── pipeline/       # Phase orchestration (each phase independent)
│       ├── config.py       # YAML config loading
│       ├── phase1_retrieval.py
│       ├── phase2_ir_aggregation.py
│       ├── phase3_reranking.py
│       ├── phase4_rerank_aggregation.py
│       ├── phase5_comparisons.py
│       └── phase6_elo.py
├── tests/              # Unit tests
├── configs/            # YAML configuration files
├── data/
│   ├── raw/                      # BEIR datasets (downloaded)
│   ├── phase1_retrieval/         # Phase 1 outputs
│   ├── phase2_ir_aggregation/    # Phase 2 outputs
│   ├── phase3_reranking/         # Phase 3 outputs
│   ├── phase4_rerank_aggregation/# Phase 4 outputs
│   ├── phase5_comparisons/       # Phase 5 outputs
│   └── phase6_rankings/          # Phase 6 outputs
└── scripts/            # Entry point scripts (one per phase)
    ├── run_phase1.py
    ├── run_phase2.py
    ├── run_phase3.py
    ├── run_phase4.py
    ├── run_phase5.py
    └── run_phase6.py
```

---

## Data Storage Schema

### File Formats

| Data Type | Format | Rationale |
|-----------|--------|-----------|
| Rankings | Parquet | Columnar, efficient for numerical data |
| Comparisons | JSONL | Append-friendly, human-readable |
| Metadata/Config | JSON | Structured, portable |
| Embeddings | Parquet or NPZ | Efficient for dense vectors |

### Directory Layout

```
data/
├── raw/
│   └── {dataset}/                      # BEIR data as-is (corpus, queries, qrels)
│
├── phase1_retrieval/
│   └── {dataset}/
│       ├── metadata.json               # Phase 1 config, timestamps, retriever versions
│       ├── bm25.parquet
│       ├── splade.parquet
│       ├── e5.parquet
│       ├── colbert.parquet
│       └── bge.parquet
│
├── phase2_ir_aggregation/
│   └── {dataset}/
│       ├── metadata.json               # Phase 2 config, which retrievers used, method params
│       ├── aggregated_rrf.parquet
│       └── aggregated_borda.parquet
│
├── phase3_reranking/
│   └── {dataset}/
│       ├── metadata.json               # Phase 3 config, k1, reranker versions
│       ├── minilm_l6.parquet
│       ├── minilm_l12.parquet
│       ├── bge_reranker.parquet
│       └── mmarco_minilm.parquet
│
├── phase4_rerank_aggregation/
│   └── {dataset}/
│       ├── metadata.json               # Phase 4 config, which rerankers used, method params
│       ├── aggregated_rrf.parquet
│       └── aggregated_borda.parquet
│
├── phase5_comparisons/
│   └── {dataset}/
│       ├── metadata.json               # Phase 5 config, model, top_k2, total pairs
│       └── {query_id}/
│           └── comparisons.jsonl       # Append-only, one comparison per line
│
└── phase6_rankings/
    └── {dataset}/
        ├── metadata.json               # Phase 6 config, ELO params
        └── {query_id}/
            ├── elo_state.json
            ├── final_ranking.parquet
            └── embeddings/
                ├── no_query.parquet
                └── with_query.parquet
```

### Parquet Schemas

**Phase 1: IR Rankings** (`phase1_retrieval/{dataset}/{retriever}.parquet`):
```
query_id: string
doc_id: string
rank: int32
score: float64
```

**Phase 2: IR Aggregated Rankings** (`phase2_ir_aggregation/{dataset}/aggregated_{method}.parquet`):
```
query_id: string
doc_id: string
rank: int32
agg_score: float64
```

**Phase 3: Reranker Rankings** (`phase3_reranking/{dataset}/{reranker}.parquet`):
```
query_id: string
doc_id: string
rank: int32
score: float64
```

**Phase 4: Reranker Aggregated Rankings** (`phase4_rerank_aggregation/{dataset}/aggregated_{method}.parquet`):
```
query_id: string
doc_id: string
rank: int32
agg_score: float64
```

**Phase 6: Final LLM Rankings** (`phase6_rankings/{dataset}/{query_id}/final_ranking.parquet`):
```
doc_id: string
rank: int32
elo_rating: float64
comparison_count: int32
```

### JSONL Schema

**Comparisons** (`comparisons.jsonl`):
```json
{
  "comparison_id": "uuid",
  "query_id": "string",
  "doc_a_id": "string",
  "doc_b_id": "string",
  "presented_order": ["doc_a_id", "doc_b_id"],
  "winner_id": "string",
  "reasoning": "string",
  "model": "string",
  "timestamp": "ISO8601"
}
```

### JSON Schema

**ELO State** (`elo_state.json`):
```json
{
  "query_id": "string",
  "k": 500,
  "initial_ratings": {"doc_id": 1500.0},
  "final_ratings": {"doc_id": 1623.4},
  "k_factor": 32,
  "iterations": 1,
  "convergence_threshold": 1e-6,
  "history": [{"iteration": 0, "ratings": {...}}]
}
```

---

## Execution Flow

Each phase is **independently executable** via its own script. Phases read from cached outputs of previous phases.

### Phase 1: IR Retrieval

```bash
python scripts/run_phase1.py --dataset scifact --retrievers bm25,splade,e5,colbert,bge
```

```
Input:  data/raw/{dataset}/
Output: data/phase1_retrieval/{dataset}/{retriever}.parquet

1. Load BEIR dataset (corpus, queries, qrels)
2. For each retriever (parallelizable across retrievers):
    a. Check if {retriever}.parquet exists (skip if complete)
    b. For each query:
        - Retrieve top-N documents with scores
    c. Write {retriever}.parquet
3. Write metadata.json (config, timestamps, versions)
```

### Phase 2: IR Rank Aggregation

```bash
python scripts/run_phase2.py --dataset scifact --methods rrf,borda
```

```
Input:  data/phase1_retrieval/{dataset}/*.parquet
Output: data/phase2_ir_aggregation/{dataset}/aggregated_{method}.parquet

1. Load all retriever parquet files from Phase 1
2. For each aggregation method:
    a. Check if aggregated_{method}.parquet exists (skip if complete)
    b. For each query:
        - Collect rankings from all retrievers
        - Compute aggregated ranking
    c. Write aggregated_{method}.parquet
3. Write metadata.json (which retrievers, method params)
```

### Phase 3: Reranker Scoring

```bash
python scripts/run_phase3.py --dataset scifact --top_k1 1000 --rerankers minilm_l6,minilm_l12,bge_reranker,mmarco_minilm
```

```
Input:  data/phase2_ir_aggregation/{dataset}/aggregated_rrf.parquet
Output: data/phase3_reranking/{dataset}/{reranker}.parquet

1. Load aggregated IR rankings from Phase 2
2. For each query:
    a. Select top-k1 documents
3. For each reranker (parallelizable across rerankers):
    a. Check if {reranker}.parquet exists (skip if complete)
    b. For each query:
        - Score all k1 query-document pairs
        - Rank by score
    c. Write {reranker}.parquet
4. Write metadata.json (config, k1, timestamps, versions)
```

### Phase 4: Reranker Rank Aggregation

```bash
python scripts/run_phase4.py --dataset scifact --methods rrf,borda
```

```
Input:  data/phase3_reranking/{dataset}/*.parquet
Output: data/phase4_rerank_aggregation/{dataset}/aggregated_{method}.parquet

1. Load all reranker parquet files from Phase 3
2. For each aggregation method:
    a. Check if aggregated_{method}.parquet exists (skip if complete)
    b. For each query:
        - Collect rankings from all rerankers
        - Compute aggregated ranking
    c. Write aggregated_{method}.parquet
3. Write metadata.json (which rerankers, method params)
```

### Phase 5: LLM Comparison Collection

```bash
python scripts/run_phase5.py --dataset scifact --query_ids q1,q2 --top_k2 500 --model llama-3.1-8b
```

```
Input:  data/phase4_rerank_aggregation/{dataset}/aggregated_rrf.parquet
Output: data/phase5_comparisons/{dataset}/{query_id}/comparisons.jsonl

1. Load aggregated reranker rankings from Phase 4
2. For each query_id:
    a. Select top-k2 documents
    b. Generate all k2*(k2-1)/2 unique pairs
    c. Load existing comparisons.jsonl (for resumability)
    d. Identify remaining pairs to compare
    e. For each remaining pair:
        - Randomize presentation order
        - Construct prompt (query + doc metadata + both docs)
        - Call LLM (reasoning enabled, constrained output)
        - Parse winner from response
        - Append to comparisons.jsonl (atomic write)
3. Write metadata.json (model, top_k2, total pairs, completed pairs)
```

### Phase 6: ELO Tournament

```bash
python scripts/run_phase6.py --dataset scifact --query_ids q1,q2
```

```
Input:  data/phase5_comparisons/{dataset}/{query_id}/comparisons.jsonl
Output: data/phase6_rankings/{dataset}/{query_id}/final_ranking.parquet
        data/phase6_rankings/{dataset}/{query_id}/elo_state.json
        data/phase6_rankings/{dataset}/{query_id}/embeddings/

1. For each query_id:
    a. Load all comparisons from Phase 5
    b. Initialize ELO ratings from reranker-aggregated ranks (or uniform)
    c. Process all comparisons through ELO updates
    d. Run tournament iterations until convergence
    e. Write elo_state.json (full history)
    f. Write final_ranking.parquet
    g. Extract document embeddings (with/without query context)
    h. Write embeddings to parquet
2. Write metadata.json (ELO params, convergence stats)
```

---

## Key Design Principles

1. **Resumability**: All operations checkpoint to disk; can resume after interruption
2. **Reproducibility**: Config files capture all parameters; random seeds logged
3. **Abstraction**: Swappable retrievers/rerankers/LLMs via common interfaces
4. **Validation**: Pydantic models enforce data integrity at boundaries
5. **Parallelism**: IR retrievers and rerankers run in parallel; LLM calls can be batched

---

## Dependencies

```
# Data & ML
beir
sentence-transformers
colbert-ai
rank-bm25
transformers
torch

# LLM
vllm
openai

# Data handling
pydantic>=2.0
pandas
pyarrow
numpy

# Config & Utils
pyyaml
tqdm

# Testing
pytest
pytest-cov
```

---

## Code Quality Standards

- Type annotate all functions and variables at time of declaration
- Annotate all arguments and return values using NumPy PyDoc style
- Aim for full unit test coverage of all mission-critical features
- Follow PEP8 throughout
- Use descriptive variable names; avoid terse and unexplained variables
- Write clean, DRY code

---

## Configuration Example

`configs/default.yaml`:
```yaml
datasets:
  - scifact
  - scidocs

retrieval:
  retrievers:
    - bm25
    - splade
    - e5
    - colbert
    - bge
  top_n: 1000  # retrieve top-N per retriever

ir_aggregation:
  methods:
    - rrf
    - borda
  rrf_k: 60

reranking:
  rerankers:
    - minilm_l6      # cross-encoder/ms-marco-MiniLM-L-6-v2
    - minilm_l12     # cross-encoder/ms-marco-MiniLM-L-12-v2
    - bge_reranker   # BAAI/bge-reranker-large
    - mmarco_minilm  # cross-encoder/mMARCO-mMiniLM-L-12-H-384-v1
  top_k1: 1000  # rerank top-k1 from IR aggregation

rerank_aggregation:
  methods:
    - rrf
    - borda
  rrf_k: 60

llm:
  small_models:
    - meta-llama/Llama-3.1-8B-Instruct
    - meta-llama/Llama-3.2-11B-Vision-Instruct
  oracle_model: gpt-4.5-turbo
  vllm:
    tensor_parallel_size: 1
    max_model_len: 8192

ranking:
  top_k2: 500  # LLM compares top-k2 from reranker aggregation
  elo:
    initial_rating: 1500
    k_factor: 32
    convergence_threshold: 1e-6

paths:
  data_dir: ./data
  raw_dir: ./data/raw
  phase1_dir: ./data/phase1_retrieval
  phase2_dir: ./data/phase2_ir_aggregation
  phase3_dir: ./data/phase3_reranking
  phase4_dir: ./data/phase4_rerank_aggregation
  phase5_dir: ./data/phase5_comparisons
  phase6_dir: ./data/phase6_rankings
```
