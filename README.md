# Zettelgraph — Local-First GraphRAG Knowledge Vault

A local-first GraphRAG knowledge management system that ingests source documents, shreds them into atomic Zettel notes via an LLM, stores notes on disk and in a LanceDB vector index, and builds a NetworkX graph for multi-hop semantic queries.

## Features

- **Multi-source ingestion**: Drop `.md`, `.txt`, `.pdf`, or `.web` files into `raw/incoming/`
- **Web parsing**: `.web` files contain URLs; each URL is fetched, converted to clean Markdown (via Readability + html2text), then shredded into zettel notes
- **Flexible LLM backends**: LM Studio (local), Ollama (local), OpenAI (cloud), Anthropic (cloud) — switch via config
- **Semantic graph queries**: Hybrid vector + multi-hop graph traversal over your note network
- **Config-driven**: All provider settings, API keys, and web fetch parameters live in `provider.yaml`

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your LLM provider (see provider.yaml.example)
cp provider.yaml.example provider.yaml
# Edit provider.yaml to set your active_provider and API keys
```

## Usage

### Ingest documents

```bash
python3 knowledge_vault.py --ingest
```

Scans `raw/incoming/`, processes each file, writes zettel notes to `zettel/`, and archives the source to `raw/archive/`.

### Query your knowledge vault

```bash
python3 knowledge_vault.py
```

Enters an interactive loop. Ask questions and get synthesized answers with citations:

```
[GraphRAG Search]: How does AI impact infrastructure?

--- Synthesized Answer ---
AI has evolved from a specialized tool to a foundational utility, comparable to electricity...
[[z_20260815_134312_7cd7]]
```

Type `exit` or `quit` to leave.

### Query using an external context file

For long context that can't be typed into the prompt, pass a file containing the context. The program synthesizes an answer directly over that file's contents instead of doing vector/graph retrieval:

```bash
python3 knowledge_vault.py --file context.txt
python3 knowledge_vault.py --file context.txt --question "Summarize the key security risks"
```

## Source File Types

| Extension | Description |
|---|---|
| `.md` / `.txt` | Plain text or Markdown — read directly |
| `.pdf` | PDF documents — text extracted via PyMuPDF |
| `.web` | Text file with one URL per line (comments start with `#`). Each URL is fetched, converted to clean Markdown, then processed as a document |

### Example `.web` file

```
# Research sources about Python
https://en.wikipedia.org/wiki/Python_(programming_language)
https://docs.python.org/3/
```

## Configuration (`provider.yaml`)

All settings are externalized to `provider.yaml`. No code changes needed to switch providers or tweak parameters.

```yaml
active_provider: OPENAI    # LMSTUDIO | OLLAMA | OPENAI | ANTHROPIC
max_tokens: 4096
temperature: 0.2

# GraphRAG token budgeting: split the model's window between retrieved
# context and the synthesized answer. Output is always reserved a minimum
# share of the window (window * output_reserve_ratio); retrieved context
# is capped at window * (1 - output_reserve_ratio).
model_context_window: 32768
output_reserve_ratio: 0.30

providers:
  LMSTUDIO:
    url: "http://localhost:1234/v1/chat/completions"
    model: "google/gemma-4-e4b"

  OLLAMA:
    url: "http://localhost:11434/api/generate"
    model: "llama3"

  OPENAI:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    api_key: ""          # or set OPENAI_API_KEY env var

  ANTHROPIC:
    url: "https://api.anthropic.com/v1/messages"
    model: "claude-3-5-sonnet-20241022"
    api_key: ""          # or set ANTHROPIC_API_KEY env var

  WEB:
    user_agent: "Mozilla/5.0 (compatible; KnowledgeVault/1.0)"
    timeout: 30
    extract_main_content: true
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `PROVIDER_CONFIG_PATH` | Override the path to `provider.yaml` |
| `OPENAI_API_KEY` | OpenAI API key (overrides yaml value) |
| `ANTHROPIC_API_KEY` | Anthropic API key (overrides yaml value) |
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | Force offline model loading |

## Project Structure

```
zettelgraph/
├── knowledge_vault.py    # Main program
├── provider.yaml         # LLM provider + web fetch configuration
├── provider.yaml.example # Template for configuration
├── requirements.txt      # Python dependencies
├── AGENTS.md             # Developer reference & code breakdown
├── manifest.json         # Ingestion dedup cache (MD5 → file metadata)
├── raw/
│   ├── incoming/         # Drop source files here
│   └── archive/          # Processed files moved here
├── zettel/               # Generated OKF Zettel notes (*.md)
└── db/
    └── zettel_lance/     # LanceDB vector database
```

## Zettel Format

Each zettel is a Markdown file with YAML frontmatter following the OKF (Open Knowledge Fragments) schema:

```markdown
---
okf_version: '1.0'
id: z_20260815_134312_7cd7
title: AI as Foundational Infrastructure
type: concept
tags: [ai-infrastructure, digital-transformation]
summary: AI is becoming as essential as electricity...
why_should_i_care: Understanding AI's role helps predict trends.
last_modified: '2026-08-15'
---

# AI as Foundational Infrastructure

> **Why should I care?** Understanding AI's role as infrastructure...

Body text with [[related_to_topic::z_20260815_134230_...]] graph links.
```

Zettel IDs follow the format: `z_{YYYYMMDD_HHMMSS}_{4-hex-chars}`.

## Developer Reference

For a detailed breakdown of the codebase — including section maps, line numbers, architecture diagrams, and data flow — see **[AGENTS.md](AGENTS.md)**.

## License

See [LICENSE](LICENSE).
