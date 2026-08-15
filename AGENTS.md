# AGENTS.md — Knowledge Vault (`knowledge_vault.py`)

A guide and reference for developers working with the Zettelgraph knowledge vault system.

---

## Program Overview

`knowledge_vault.py` is a local-first GraphRAG knowledge management system. It ingests source documents (`.md`, `.txt`, `.pdf`, and `.web` files containing URLs), shreds them into atomic Zettel notes via an LLM, stores the notes on disk and in a LanceDB vector index, and builds a NetworkX directed graph for multi-hop semantic queries.

### Entry Points

| Invocation | Behavior |
|---|---|
| `python3 knowledge_vault.py --ingest` | Runs `process_ingestion_pipeline()` — scans `raw/incoming/`, processes new files, writes zettels, updates the graph cache |
| `python3 knowledge_vault.py` (no args) | Launches an interactive GraphRAG query loop — prompts for questions, traverses the graph, and synthesizes answers via the configured LLM |

---

## Architecture / Component Map

```
┌─────────────────────────────────────────────────────────┐
│                     provider.yaml                        │
│  (Active provider, API keys, model names, WEB config)   │
└───────────────┬────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│           CONFIG LOADING  (lines 24–65)                  │
│  Reads provider.yaml → sets PROVIDER, LLM params,        │
│  web fetch settings (WEB_USER_AGENT, WEB_TIMEOUT)        │
└───────────────┬────────────────────────────────────────┘
                │
     ┌──────────┴───────────┬──────────────┐
     ▼                      ▼              ▼
┌──────────┐  ┌──────────────────┐  ┌───────────────┐
│INgestion │  │LLM PROVIDERS     │  │Web Fetching   │
│Pipeline  │  │(LMStudio, OpenAI,│  │(requests +    │
│          │  │  Anthropic,      │  │ readability + │
│          │  │  Ollama)         │  │ html2text)    │
└────┬─────┘  └────────┬─────────┘  └───────┬───────┘
     │                 │                  │
     │                 ▼                  │
     │          ┌──────────────┐          │
     │          │  _shred_text │          │
     │          │  _to_zettels │          │
     │          │  (chunking + │          │
     │          │   LLM call + │          │
     │          │   parsing)    │          │
     │          └──────┬───────┘          │
     │                 │                  │
     │                 ▼                  │
     │        ┌──────────────────┐        │
     │        │  OKFZettel       │        │
     │        │  .write_to_disk  │        │
     │        │  (writes .md)    │        │
     │        └────────┬─────────┘        │
     │                 │                  │
     ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│  LanceDB (embeddings)    raw/archive/  (archived files) │
│  zettel/ (OKF .md notes)  db/zettel_lance/ (vectors)    │
│  manifest.json (dedup)    .graph_cache.json (topology)   │
└─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│        LocalGraphRAGRouter  (query layer)                │
│  rebuild_graph_from_vault() → cache or cold-start      │
│  query(question) → vector search + ego graph traversal  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Sections (by line number)

### Configuration Loading — Lines 24–65

- **`provider.yaml` path**: `PROVIDER_CONFIG_PATH` env var, defaults to `provider.yaml`
- **Global settings**: `PROVIDER`, `MAX_TOKENS`, `TEMPERATURE`
- **Per-provider config**: `LMSTUDIO_*`, `OLLAMA_*`, `OPENAI_*`, `ANTHROPIC_*` — each reads from the `providers:` block in `provider.yaml` with inline fallbacks. Cloud API keys prefer environment variables over YAML values.
- **Web config**: `WEB_USER_AGENT`, `WEB_TIMEOUT`, `WEB_EXTRACT_MAIN_CONTENT` — read from the `WEB:` section of `provider.yaml`
- **Path constants**: `RAW_INCOMING`, `RAW_ARCHIVE`, `ZETTEL_DIR`, `DB_URI`, `MANIFEST_FILE`

### OKFZettel Class — Lines 69–119

Handles OKF-compliant Zettel note parsing and serialization.

- **`LINK_PATTERN`**: Regex `\[\[([a-zA-Z0-9_\-]+)::([a-zA-Z0-9_\-]+)\]\]` — matches Zettel relationship links
- **`__init__(file_path)`**: Loads and parses a `.md` file if it exists
- **`load_and_parse()`**: Extracts YAML frontmatter metadata block (`---\n...---\n`), stores the remaining body as `content_body`, and scrapes inline `[[relation::target_id]]` links into `self.edges`
- **`write_to_disk()`** (classmethod): Writes a new Zettel with a YAML frontmatter header (okf_version, id, title, type, tags, summary, why_should_i_care, last_modified), optional "Why should I care?" blockquote, and body. Returns the reloaded `OKFZettel` instance.

  **Zettel ID format**: `z_{YYYYMMDD_HHMMSS}_{4-hex-chars}` (e.g., `z_20260815_134312_7cd7`)

### LLM Interaction Utilities — Lines 121–223

- **`call_llm(prompt, system_prompt)`**: Dispatches to the active provider’s handler. Returns `None` on unknown provider or failure.
- **`_call_lmstudio`**: POSTs to `LMSTUDIO_URL` (OpenAI-compatible `/chat/completions`). Uses `_extract_openai_response`.
- **`_call_openai`**: POSTs to `{OPENAI_BASE_URL}/chat/completions` with Bearer auth. Uses `_extract_openai_response`.
- **`_call_anthropic`**: POSTs to `ANTHROPIC_URL` (`/v1/messages`) with `x-api-key` header. Uses `_extract_anthropic_response`.
- **`_call_ollama`**: POSTs to `OLLAMA_URL` (`/api/generate`) with `stream: False`. Uses `_extract_ollama_response`.
- **`_request(url, headers, payload, extract_fn)`**: Shared HTTP POST helper. 120s timeout. Checks for 200 status, parses JSON, extracts text via the passed `extract_fn`. Debug-prints empty responses.
- **`_extract_*_response`**: Response unwrappers for each provider’s JSON schema.
- **`extract_text(path)`**: File-type dispatcher. `.md`/`.txt` → raw read. `.pdf` → PyMuPDF page-by-page extraction. Returns `None` for unrecognized extensions.

### Web Content Helpers — Lines 236–276

- **`convert_url_to_markdown(url)`**: Fetches HTML via `requests` (with `WEB_USER_AGENT` and `WEB_TIMEOUT`), optionally runs Readability (`Document`) to extract the main article content and title, then converts HTML → Markdown via `html2text` (images ignored, no line wrapping). Prepends `# {title}` if a title was extracted. Returns `None` on fetch error.
- **`extract_urls_from_web_file(path)`**: Reads a `.web` text file, returns a list of URLs (one per line, skipping `#` comments and blanks).

### Ingestion Pipeline — Lines 278–427

#### `_shred_text_to_zettels(raw_text, source_label, table, embed_model)` — Lines 278–370

Extracted helper that runs the full LLM-to-Zettel pipeline on a block of text:

1. **Chunking**: Splits text into ~100KB chunks with 500-char overlap.
2. **Per-chunk LLM call**: Sends each chunk to `call_llm` with a system prompt that asks the LLM to output atomic zettel concepts using `<<<START_ZETTEL>>>` / `<<<END_ZETTEL>>>` delimiters with TITLE, TYPE, TAGS, SUMMARY, WHY, RELATED, and BODY fields.
3. **Parsing**: Extracts zettel blocks with regex, then scrapes each field via `get_field()` (multiline regex).
4. **Vector link resolution**: For each RELATED topic, encodes it with `embed_model`, searches LanceDB for the nearest neighbor, and creates a `[[related_to_{topic}::{target_id}]]` link.
5. **Writing**: Calls `OKFZettel.write_to_disk()` to write the `.md` file, then adds the zettel body’s embedding to LanceDB (`table.add`).

Called by both `.web` URLs (markdown from `convert_url_to_markdown`) and regular files (text from `extract_text`).

#### `process_ingestion_pipeline()` — Lines 372–427

Top-level ingestion orchestrator:

1. Connects to LanceDB (`DB_URI`), opens/creates the `notes` table.
2. Loads `SentenceTransformer('all-MiniLM-L6-v2')` embed model.
3. Loads `manifest.json` for dedup (MD5 hash of source file content).
4. Ensures `raw/incoming/`, `raw/archive/`, and `zettel/` directories exist.
5. **Per-file processing loop** (lines 289–415):
   - Skips directories and dotfiles.
   - MD5-hashes the file; skips if already in manifest.
   - **If `.web` file**: calls `extract_urls_from_web_file` → for each URL calls `convert_url_to_markdown` → passes result to `_shred_text_to_zettels`.
   - **Otherwise**: calls `extract_text` → passes result to `_shred_text_to_zettels`.
   - Skips content under 100 chars.
   - After processing: records hash in manifest, renames source file to `raw/archive/`.
6. Writes updated manifest to `manifest.json`.
7. **Graph cache rebuild**: Deletes stale `.graph_cache.json`, calls `LocalGraphRAGRouter.rebuild_graph_from_vault()`.

### LocalGraphRAGRouter Class — Lines 429–558

Compiles flat zettel files into a searchable NetworkX `DiGraph`.

- **`__init__(zettel_dir, db_uri)`**: Connects to LanceDB, loads the embed model, initializes empty graph + vault.
- **`rebuild_graph_from_vault()`** (lines 439–486): Two-phase build:
  1. **Cache check**: If `.graph_cache.json` exists and is valid, load nodes + edges from it (instant).
  2. **Cold-start fallback**: Scans `zettel/*.md`, creates `OKFZettel` objects, populates graph nodes with metadata, then injects `[[relation::target]]` edges from each zettel’s parsed `self.edges`.
  3. Saves the cache via `save_graph_cache()`.
- **`save_graph_cache()`** (lines 488–503): Serializes the graph to `.graph_cache.json` with `nodes` (ID → metadata) and `edges` (source, target, relation) arrays.
- **`query(question, max_hops=2)`** (lines 505–558): Hybrid search:
  1. Rebuilds graph from cache.
  2. Encodes the question → vector search in LanceDB, limits to 3 results.
  3. Cross-references vector results with graph nodes to find seed nodes.
  4. Builds an ego network (multi-hop traversal, radius = `max_hops`) around each seed.
  5. Assembles context: relational pathways + full zettel payloads (re-read from disk).
  6. Synthesizes an answer via `call_llm` with a "Thinking Partner" system prompt that cites node IDs as `[[z_id]]`.

### Interactive Query Loop — Lines 560–572

When run without `--ingest`, enters a `while True` loop prompting for queries, calling `router.query()`, and printing synthesized answers. Exits on `exit` / `quit`.

---

## Data Flow

### Ingestion (`--ingest`)
```
raw/incoming/ ──→ manifest check (MD5 hash) ──→ extract_text() OR (.web → fetch URLs → convert_url_to_markdown)
                                                       │
                                                       ▼
                                          _shred_text_to_zettels()
                                          │  chunk → LLM → parse zettels
                                          │  → OKFZettel.write_to_disk()
                                          │  → LanceDB table.add()
                                          ▼
                                   zettel/*.md  +  db/zettel_lance/  +  manifest.json
                                                       │
                                                       ▼
                                          archive file → raw/archive/
```

### Query (interactive)
```
User question → embed → LanceDB vector search → seed nodes
                       → ego graph (max_hops) → context assembly
                       → LLM synthesis → answer with [[z_id]] citations
```

---

## Configuration Reference (`provider.yaml`)

```yaml
active_provider: OPENAI | LMSTUDIO | OLLAMA | ANTHROPIC
max_tokens: 4096
temperature: 0.2

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
    api_key: ""        # or set OPENAI_API_KEY env var

  ANTHROPIC:
    url: "https://api.anthropic.com/v1/messages"
    model: "claude-3-5-sonnet-20241022"
    api_key: ""        # or set ANTHROPIC_API_KEY env var

  WEB:                 # web content fetching settings
    user_agent: "Mozilla/5.0 (compatible; KnowledgeVault/1.0; ...)"
    timeout: 30
    extract_main_content: true
```

---

## Directory Layout

```
zettelgraph/
├── knowledge_vault.py        # Main program (see AGENTS.md for section map)
├── provider.yaml             # LLM provider + web fetch configuration
├── requirements.txt          # Python dependencies
├── manifest.json             # Ingestion dedup cache (MD5 → file metadata)
├── raw/
│   ├── incoming/             # Drop source files here (.md, .txt, .pdf, .web)
│   └── archive/              # Processed files are moved here
├── zettel/                   # Generated OKF Zettel notes (*.md)
├── db/
│   └── zettel_lance/         # LanceDB vector database (embeddings + index)
└── __pycache__/
```

---

## Zettel File Format (OKF)

Each zettel is a Markdown file with YAML frontmatter:

```markdown
---
okf_version: '1.0'
id: z_20260815_134312_7cd7
title: AI as Foundational Infrastructure
type: concept
tags:
- ai-infrastructure
- digital-transformation
summary: AI is becoming as essential as electricity...
why_should_i_care: Understanding AI's role as infrastructure...
last_modified: '2026-08-15'
---

# AI as Foundational Infrastructure

> **Why should I care?** Understanding AI's role as infrastructure...

Body text with [[related_to_topic::z_20260815_...]] graph links.
```

### Relationship Links

Inline links use the format `[[<relation>::<target_zettel_id>]]`. The `relation` is a slugified version of the RELATED topic from the LLM output. These are parsed by `OKFZettel.LINK_PATTERN` and turned into directed edges in the NetworkX graph.

---

## Web Source Files (`.web`)

A `.web` file is a plain text list of URLs to fetch. Format:

```
# Comments start with #
https://en.wikipedia.org/wiki/Python_(programming_language)
https://example.com/article
```

Processing flow:
1. `extract_urls_from_web_file()` reads URLs (non-comment, non-blank lines).
2. For each URL: `convert_url_to_markdown()` fetches the page, optionally extracts the main content via Readability, and converts HTML → Markdown via `html2text`.
3. The resulting Markdown is fed to `_shred_text_to_zettels()` — identical to how regular file content is processed.
4. The `.web` file itself is hashed for manifest dedup and archived after all URLs are processed.

---

## Utility Commands

| Command | Purpose |
|---|---|
| `python3 knowledge_vault.py --ingest` | Run the full ingestion pipeline |
| `python3 knowledge_vault.py` | Launch interactive query loop |
| `python3 -c "from knowledge_vault import LocalGraphRAGRouter; LocalGraphRAGRouter('./zettel', './db/zettel_lance').rebuild_graph_from_vault(); print('...cache built...')"` | Force-rebuild the graph cache from disk |
| `rm -f zettel/.graph_cache.json` | Clear the graph cache to force a cold-start rebuild on next query |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `PROVIDER_CONFIG_PATH` | Override the path to `provider.yaml` (defaults to `provider.yaml`) |
| `OPENAI_API_KEY` | OpenAI API key (overrides `provider.yaml` value) |
| `ANTHROPIC_API_KEY` | Anthropic API key (overrides `provider.yaml` value) |
| `HF_HUB_OFFLINE=1` | Forces Hugging Face models to load from local cache (set at import) |
| `TRANSFORMERS_OFFLINE=1` | Same offline behavior for Transformers (set at import) |
