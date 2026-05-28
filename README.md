# BPSD Clinical Evidence Audit RAG System

## Overview

This project is a retrieval-augmented generation (RAG) system for auditing behavioral and psychological symptom evidence from unstructured clinical notes.

The application retrieves semantically relevant evidence snippets from embedded clinical note units using hybrid retrieval (embedding similarity + keyword matching), then uses a large language model (LLM) to determine whether evidence exists for a target symptom/domain.

The system was originally developed for Behavioral and Psychological Symptoms of Dementia (BPSD) phenotyping workflows using Bio_ClinicalBERT embeddings and clinical note text, but the architecture is modular and can support other clinical concept auditing tasks.

The app supports multiple deployment backends:

* Databricks + Spark + Databricks Model Serving
* Local CSV-based testing
* Dockerized deployment
* OpenAI-compatible LLM APIs

---

# Features

## Hybrid Retrieval

The system combines:

### Semantic Retrieval

* Uses cosine similarity between:

  * note-unit embeddings
  * symptom-query embeddings
* Retrieves top semantic matches per query term

### Keyword Retrieval

* Performs rule-based keyword matching
* Supplements embedding retrieval
* Helps recover exact lexical matches

### Evidence Consolidation

* Deduplicates overlapping retrieval results
* Merges semantic and keyword evidence
* Preserves note ordering

---

# LLM-Based Evidence Extraction

After retrieval, the system sends candidate evidence snippets to an LLM and asks it to determine:

* whether evidence is present
* whether evidence is negated
* whether evidence is historical
* the exact supporting excerpt

The model returns structured JSON output.

Example:

```json
{
  "target": "agitation",
  "evidence_present": "yes",
  "historical": "no",
  "negated": "no",
  "evidence_excerpt": "Patient increasingly agitated overnight."
}
```

---

# Architecture

## High-Level Flow

```text
User enters ClinicalNoteKey
            ↓
Fetch note embedding units
            ↓
Fetch symptom query embeddings
            ↓
Hybrid retrieval
  ├── semantic similarity
  └── keyword matching
            ↓
Merge + rank evidence units
            ↓
Build evidence prompt
            ↓
LLM JSON extraction
            ↓
Display audit results
```

---

# Repository Structure

```text
bpsd_audit_app/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── config_loader.py
│
├── configs/
│   ├── local.yaml
│   └── databricks.yaml
│
├── connectors/
│   ├── base.py
│   ├── factory.py
│   ├── databricks.py
│   └── local_files.py
│
├── core/
│   ├── retrieval.py
│   ├── llm.py
│   └── audit.py
│
└── data/
    ├── note_embeddings.csv
    └── symptom_embeddings.csv
```

---

# Connector-Based Design

The application separates:

* retrieval logic
* data access
* LLM provider
* UI

This allows the same retrieval pipeline to run against different backends.

## Supported Connectors

| Connector           | Data Source  |
| ------------------- | ------------ |
| DatabricksConnector | Spark tables |
| LocalFileConnector  | CSV files    |

Future connectors can be added for:

* Postgres
* Snowflake
* S3 parquet
* Databricks SQL Warehouse
* Vector databases

---

# Data Format

## Note Embedding CSV

Expected columns:

```text
UnitRowId
ClinicalNoteKey
EncounterKey
PatientDurableKey
Section
UnitId
UnitText
UnitTextEmbedding
UnitTextEmbeddingNorm
```

Example embedding format:

```csv
"[0.123, 0.456, 0.789]"
```

---

## Symptom Embedding CSV

Expected columns:

```text
Category
TermIdx
QueryText
QueryTextEmbedding
QueryTextEmbeddingNorm
```

---

# Running Locally

## 1. Install Docker

Install Docker Desktop:

https://www.docker.com/products/docker-desktop/

Verify installation:

```bash
docker --version
```

---

# 2. Clone Repository

```bash
git clone <repo_url>
cd bpsd_audit_app
```

---

# 3. Add CSV Files

Place CSV files inside:

```text
data/
```

Example:

```text
data/
  note_embeddings.csv
  symptom_embeddings.csv
```

---

# 4. Create OpenAI API Key

Create an API key:

https://platform.openai.com/api-keys

---

# 5. Build Docker Image

```bash
docker build -t bpsd-audit-app .
```

---

# 6. Run Local Mode

```bash
docker run \
  -p 8501:8501 \
  -e APP_ENV=local \
  -e OPENAI_API_KEY="your_key_here" \
  -v $(pwd)/data:/app/data \
  bpsd-audit-app
```

---

# 7. Open App

Open browser:

```text
http://localhost:8501
```

---

# Running Databricks Mode

## Configure Environment

Set:

```bash
APP_ENV=databricks
```

and provide:

```bash
DATABRICKS_TOKEN
```

Example:

```bash
docker run \
  -p 8501:8501 \
  -e APP_ENV=databricks \
  -e DATABRICKS_TOKEN="your_token" \
  bpsd-audit-app
```

---

# Configuration System

The application loads deployment-specific configuration files.

## Example

### local.yaml

```yaml
data:
  connector: local_files
  note_embeddings_path: data/note_embeddings.csv
  symptom_embeddings_path: data/symptom_embeddings.csv

llm:
  provider: openai
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
```

### databricks.yaml

```yaml
data:
  connector: databricks

llm:
  provider: databricks
  model: databricks-gpt-5-4-mini
```

---

# Retrieval Pipeline

## Semantic Retrieval

For each:

* note embedding
* symptom query embedding

the system computes cosine similarity:

```python
similarity = dot(a, b) / (||a|| * ||b||)
```

Top matching units are retained.

---

# Keyword Retrieval

The system also performs rule-based matching using domain-specific symptom keywords.

This helps recover:

* lexical mentions
* exact phrase matches
* low-similarity but clinically relevant snippets

---

# Evidence Aggregation

Retrieved evidence units are:

* deduplicated
* merged across retrieval methods
* sorted by note order
* formatted into an LLM evidence block

---

# LLM Prompting

The LLM receives:

* symptom/domain
* retrieved evidence snippets

and returns structured JSON describing:

* evidence presence
* negation
* historical context
* supporting excerpt


---

# Disclaimer

This tool is intended for research and development purposes only and is not approved for clinical decision-making.
