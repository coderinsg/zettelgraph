import os
import re
import uuid
import json
import yaml
import hashlib
import requests
import lancedb
import logging
import fitz  # PyMuPDF
import networkx as nx
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import html2text
from readability import Document

# 1. Silence all Hugging Face Hub connection warnings and logger noise
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIGURATION (loaded from provider.yaml) ---
_CONFIG_PATH = os.environ.get("PROVIDER_CONFIG_PATH", "provider.yaml")
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
        _CONFIG = yaml.safe_load(_f) or {}
else:
    print(f"      ! provider.yaml not found at {_CONFIG_PATH}; falling back to built-in defaults.")
    _CONFIG = {}

PROVIDER = _CONFIG.get("active_provider", "LMSTUDIO").upper()
MAX_TOKENS = int(_CONFIG.get("max_tokens", 4096))
TEMPERATURE = float(_CONFIG.get("temperature", 0.2))

# Context/output budgeting — keep the generated answer well-fed by reserving a
# share of the model's window for output and capping how much graphite context
# is allowed to consume. Both are externalized so they can be tuned per model.
MODEL_CONTEXT_WINDOW = int(_CONFIG.get("model_context_window", 16384))
OUTPUT_RESERVE_RATIO = float(_CONFIG.get("output_reserve_ratio", 0.30))

# Reserve at least RESERVED_OUTPUT_TOKENS for generation, even if max_tokens is
# configured lower; effectives cap is the larger of the two. Everything left
# over is the hard token budget the GraphRAG context must fit within.
RESERVED_OUTPUT_TOKENS = int(MODEL_CONTEXT_WINDOW * OUTPUT_RESERVE_RATIO)
EFFECTIVE_MAX_TOKENS = max(MAX_TOKENS, RESERVED_OUTPUT_TOKENS)
CONTEXT_TOKEN_BUDGET = max(1, MODEL_CONTEXT_WINDOW - EFFECTIVE_MAX_TOKENS)

# Rough chars-per-token heuristic used to estimate context size without running a
# tokenizer every query — close enough to keep the window under control.
# The client-side tokenizer can be consulted for higher precision if ever needed.
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    """Approximates token count from character length (~4 chars/token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)

_P = _CONFIG.get("providers", {})

# Per-provider connection parameters — each resolves from provider.yaml with inline fallbacks
LMSTUDIO_URL = _P.get("LMSTUDIO", {}).get("url", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL = _P.get("LMSTUDIO", {}).get("model", "google/gemma-4-e4b")

OLLAMA_URL = _P.get("OLLAMA", {}).get("url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _P.get("OLLAMA", {}).get("model", "llama3")

OPENAI_BASE_URL = _P.get("OPENAI", {}).get("base_url", "https://api.openai.com/v1")
OPENAI_MODEL = _P.get("OPENAI", {}).get("model", "gpt-4o")
# Cloud API keys: env var takes precedence, then provider.yaml value
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or _P.get("OPENAI", {}).get("api_key", "")

ANTHROPIC_URL = _P.get("ANTHROPIC", {}).get("url", "https://api.anthropic.com/v1/messages")
ANTHROPIC_MODEL = _P.get("ANTHROPIC", {}).get("model", "claude-3-5-sonnet-20241022")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or _P.get("ANTHROPIC", {}).get("api_key", "")

# Web Content Fetching Configuration (loaded from provider.yaml)
WEB_USER_AGENT = _P.get("WEB", {}).get("user_agent", "Mozilla/5.0 (compatible; KnowledgeVault/1.0)")
WEB_TIMEOUT = int(_P.get("WEB", {}).get("timeout", 30))
WEB_EXTRACT_MAIN_CONTENT = _P.get("WEB", {}).get("extract_main_content", True)

RAW_INCOMING = "./raw/incoming"
RAW_ARCHIVE = "./raw/archive"
ZETTEL_DIR = "./zettel"
DB_URI = "./db/zettel_lance"
MANIFEST_FILE = "manifest.json"

# --- CORE GRAPH & STORAGE ENGINE ---

class OKFZettel:
    """Handles the parsing, verification, and serialization of OKF-compliant Zettel notes."""
    LINK_PATTERN = re.compile(r'\[\[([a-zA-Z0-9_\-]+)::([a-zA-Z0-9_\-]+)\]\]')

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.metadata: Dict = {}
        self.content_body: str = ""
        self.edges: List[Tuple[str, str]] = []
        if self.file_path.exists():
            self.load_and_parse()

    def load_and_parse(self):
        """Extracts strict OKF structural metadata blocks and graphs inline links."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL | re.MULTILINE)
        if yaml_match:
            self.metadata = yaml.safe_load(yaml_match.group(1)) or {}
            self.content_body = content[yaml_match.end():].strip()
        else:
            self.content_body = content.strip()
            self.metadata = {"id": self.file_path.stem, "title": self.file_path.stem, "type": "concept", "tags": [], "summary": "", "why_should_i_care": ""}

        self.edges = self.LINK_PATTERN.findall(self.content_body)

    @classmethod
    def write_to_disk(cls, directory: Path, z_id: str, title: str, z_type: str,
                      summary: str, tags: List[str], body: str, why_care: str = "") -> 'OKFZettel':
        """Enforces clean schema verification before writing files to local disk storage."""
        file_path = directory / f"{z_id}.md"
        metadata = {
            "okf_version": "1.0",
            "id": z_id,
            "title": title,
            "type": z_type,
            "tags": tags,
            "summary": summary,
            "why_should_i_care": why_care,
            "last_modified": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }
        yaml_header = yaml.dump(metadata, sort_keys=False, allow_unicode=True)
        body = body.strip()
        if why_care:
            body = f"> **Why should I care?** {why_care.strip()}\n\n{body}"
        full_markdown = f"---\n{yaml_header}---\n\n# {title}\n\n{body}\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
        return cls(file_path)

# --- UTILITIES & LLM INTERACTION ---

def call_llm(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Routes LLM requests to the configured provider, supporting local and cloud endpoints."""
    handlers = {
        "LMSTUDIO": _call_lmstudio,
        "OLLAMA": _call_ollama,
        "OPENAI": _call_openai,
        "ANTHROPIC": _call_anthropic,
    }
    handler = handlers.get(PROVIDER.upper())
    if not handler:
        print(f"      ! Unknown LLM provider configured: {PROVIDER}. Valid: LMSTUDIO, OLLAMA, OPENAI, ANTHROPIC")
        return None
    return handler(prompt, system_prompt)

def _call_lmstudio(prompt: str, system_prompt: str) -> Optional[str]:
    """Sends a chat completion request to a local LM Studio (OpenAI-compatible) endpoint."""
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": EFFECTIVE_MAX_TOKENS,
    }
    return _request(LMSTUDIO_URL, {"Content-Type": "application/json"}, payload, _extract_openai_response)

def _call_openai(prompt: str, system_prompt: str) -> Optional[str]:
    """Sends a chat completion request to the OpenAI cloud API (OpenAI-compatible endpoint)."""
    if not OPENAI_API_KEY:
        print("      ! OPENAI_API_KEY environment variable is not set. Cannot reach OpenAI endpoints.")
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": EFFECTIVE_MAX_TOKENS,
    }
    return _request(f"{OPENAI_BASE_URL}/chat/completions", headers, payload, _extract_openai_response)

def _call_anthropic(prompt: str, system_prompt: str) -> Optional[str]:
    """Sends a message request to the Anthropic cloud API (Claude models)."""
    if not ANTHROPIC_API_KEY:
        print("      ! ANTHROPIC_API_KEY environment variable is not set. Cannot reach Anthropic endpoints.")
        return None
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-07",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": EFFECTIVE_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt
    return _request(ANTHROPIC_URL, headers, payload, _extract_anthropic_response)

def _call_ollama(prompt: str, system_prompt: str) -> Optional[str]:
    """Sends a generation request to a local Ollama endpoint."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "system": system_prompt, "stream": False}
    return _request(OLLAMA_URL, {"Content-Type": "application/json"}, payload, _extract_ollama_response)

def _request(url: str, headers: Dict, payload: Dict, extract_fn) -> Optional[str]:
    """Dispatches an HTTP POST to an LLM endpoint and unwraps the generated text."""
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if response.status_code != 200:
            print(f"      ! HTTP Error: Server responded with status {response.status_code}")
            return None

        res_obj = response.json()
        raw_output = extract_fn(res_obj)

        if not raw_output or not raw_output.strip():
            # Detect truncated generations (hit the token ceiling) to give an actionable message
            finish_reason = None
            try:
                finish_reason = res_obj['choices'][0].get('finish_reason') or res_obj['choices'][0].get('native_finish_reason')
            except Exception:
                pass
            if finish_reason in ('length', 'max_tokens'):
                print(f"      ! LLM response truncated: hit max_tokens ({EFFECTIVE_MAX_TOKENS}) before producing content.")
            else:
                print(f"\n[DEBUG] Empty response generated. Raw Server JSON Payload:\n{res_obj}\n")
            return None

        return raw_output
    except Exception as e:
        print(f"      ! Connection Failure: {e}")
        return None

def _extract_openai_response(res_obj: Dict) -> str:
    """Extracts text from an OpenAI-compatible (incl. LM Studio) response."""
    return res_obj.get('choices', [{}])[0].get('message', {}).get('content', '')

def _extract_ollama_response(res_obj: Dict) -> str:
    """Extracts text from an Ollama generate response."""
    return res_obj.get('response', '')

def _extract_anthropic_response(res_obj: Dict) -> str:
    """Extracts text from an Anthropic messages response."""
    content = res_obj.get('content', [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block.get('text', '')
    return ''

def extract_text(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.md', '.txt']:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    elif ext == '.pdf':
        text = ""
        with fitz.open(path) as doc:
            for page in doc: text += page.get_text("text") + "\n"
        return text.strip()
    return None

def convert_url_to_markdown(url: str) -> Optional[str]:
    """Fetches a URL and converts the HTML content to clean Markdown."""
    try:
        headers = {"User-Agent": WEB_USER_AGENT}
        response = requests.get(url, headers=headers, timeout=WEB_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"      ! Failed to fetch {url}: {e}")
        return None

    html_content = response.text

    if WEB_EXTRACT_MAIN_CONTENT:
        try:
            doc = Document(html_content)
            title = doc.title() or ""
            html_content = doc.summary()
        except Exception:
            title = ""
    else:
        title = ""

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.body_width = 0
    markdown = h.handle(html_content)

    if title:
        markdown = f"# {title}\n\n{markdown}"

    return markdown.strip()

def extract_urls_from_web_file(path: str) -> List[str]:
    """Reads a .web source file and returns a list of URLs (one per non-comment line)."""
    urls = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls

def _shred_text_to_zettels(raw_text: str, source_label: str, table, embed_model) -> None:
    """Chunks text and runs the LLM pipeline to produce OKF zettel notes."""
    chunk_size, overlap = 102400, 500
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size - overlap)]
    print(f"    > Shredding [{source_label}] into {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks):
        print(f"      Processing chunk {idx + 1}/{len(chunks)}...")

        shred_sys = (
            "Deconstruct the incoming text into atomic, self-contained zettel concepts.\n"
            "Output each concept using the plain-text layout below.\n\n"
            "<<<START_ZETTEL>>>\n"
            "TITLE: Title of the concept\n"
            "TYPE: concept\n"
            "TAGS: tag1, tag2\n"
            "SUMMARY: Short structural overview under 50 words.\n"
            "WHY: One sentence. Why a reader should care about this concept.\n"
            "RELATED: Key Topic A, Key Topic B\n"
            "BODY:\n"
            "Deep-dive analytical description. Minimum 200 words.\n"
            "Explain mechanics and implications.\n"
            "<<<END_ZETTEL>>>"
        )

        raw_output = call_llm(f"TARGET CONTENT TO TRANSLATE:\n{chunk}", shred_sys)
        if not raw_output:
            print(f"      ! Chunk {idx + 1} failed: The LLM returned a blank string. Retrying with smaller sub-chunks.")
            # A blank response usually means the provider hit its token ceiling on a large chunk.
            # Retry the same content as smaller sub-chunks so it actually gets processed.
            sub_chunk_size = chunk_size // 2
            sub_chunks = [chunk[i:i + sub_chunk_size] for i in range(0, len(chunk), sub_chunk_size - overlap)]
            for sub_idx, sub_chunk in enumerate(sub_chunks):
                print(f"        Processing sub-chunk {sub_idx + 1}/{len(sub_chunks)}...")
                sub_output = call_llm(f"TARGET CONTENT TO TRANSLATE:\n{sub_chunk}", shred_sys)
                if not sub_output:
                    print(f"        ! Sub-chunk {sub_idx + 1} failed: The LLM returned a blank string.")
                    continue
                _process_llm_output(sub_output, idx, sub_chunk, table, embed_model, source_label)
            continue

        # Parse out zettel text blocks cleanly using text tokens
        _process_llm_output(raw_output, idx, chunk, table, embed_model, source_label)

def _process_llm_output(raw_output: str, chunk_idx: int, chunk_text: str, table, embed_model, source_label: str) -> None:
    """Parses a raw LLM response into zettel blocks and persists each one."""
    zettel_blocks = re.findall(r'<<<START_ZETTEL>>>(.*?)<<<END_ZETTEL>>>', raw_output, re.DOTALL)

    if not zettel_blocks:
        print(f"      ! Chunk {chunk_idx + 1} skipped: Could not locate text bounds (<<<START_ZETTEL>>>).")
        return

    for block in zettel_blocks:
        # Direct line-by-line field scrapers
        def get_field(field_name: str) -> str:
            m = re.search(rf'^\s*{field_name}\s*:\s*(.*)$', block, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        title_val = get_field("TITLE")
        type_val = get_field("TYPE") or "concept"
        summary_val = get_field("SUMMARY")
        why_val = get_field("WHY")
        related_raw = get_field("RELATED")

        tags_raw = get_field("TAGS")
        tags_val = [t.strip() for t in tags_raw.split(',') if t.strip()]

        body_match = re.search(r'BODY:\s*(.*)', block, re.DOTALL | re.IGNORECASE)
        body_val = body_match.group(1).strip() if body_match else ""

        if not title_val or not body_val: continue

        z_id = f"z_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        # Deterministic Vector Link Resolution using LanceDB
        discovered_links = []
        related_topics = [r.strip() for r in related_raw.split(',') if r.strip()]
        for topic in related_topics:
            topic_vector = embed_model.encode(topic).tolist()
            neighbors = table.search(topic_vector).limit(1).to_list()
            if neighbors and neighbors[0]['id'] != 'init':
                target_id = neighbors[0]['id']
                rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower().strip())
                discovered_links.append(f"[[related_to_{rel_type}::{target_id}]]")

        # Combine body with discovered relational links
        if discovered_links:
            link_section = "\n".join([f"- {link}" for link in discovered_links])
            final_body = f"{body_val}\n\n## Graph Connections\n{link_section}"
        else:
            final_body = body_val

        # Write out pure Markdown files to disk
        OKFZettel.write_to_disk(
            directory=Path(ZETTEL_DIR),
            z_id=z_id,
            title=title_val,
            z_type=type_val,
            summary=summary_val,
            tags=tags_val,
            body=final_body,
            why_care=why_val
        )

        # Track position updates inside LanceDB index
        embedding = embed_model.encode(body_val).tolist()
        table.add([{"vector": embedding, "id": z_id, "text": f"{title_val}: {summary_val}"}])
        print(f"      + Successfully Ingested OKF Zettel: {z_id} ({title_val[:40]}...)")


def process_ingestion_pipeline():
    """Builds atomic OKF nodes using plain text structural tagging rather than fragile JSON schemas."""
    print("Initializing Embedding Model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embed_dim = embed_model.get_sentence_embedding_dimension()

    print("Initializing Database Connections...")
    db = lancedb.connect(DB_URI)
    try:
        table = db.open_table("notes")
    except Exception:
        table = db.create_table("notes", data=[{"vector": [0.0]*embed_dim, "id": "init", "text": "root"}])
    manifest = {}
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f: manifest = json.load(f)

    for folder in [RAW_INCOMING, RAW_ARCHIVE, ZETTEL_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(RAW_INCOMING):
        path = os.path.join(RAW_INCOMING, filename)
        if os.path.isdir(path) or filename.startswith('.'): continue

        hasher = hashlib.md5()
        with open(path, 'rb') as f: hasher.update(f.read())
        f_hash = hasher.hexdigest()
        if f_hash in manifest: continue

        print(f"\n>>> Processing Source File: {filename}")

        if filename.lower().endswith('.web'):
            urls = extract_urls_from_web_file(path)
            for url in urls:
                print(f"    > Fetching URL: {url}")
                md_text = convert_url_to_markdown(url)
                if not md_text or len(md_text) < 100:
                    print(f"      ! Skipped: Retrieved content too short or empty.")
                    continue
                _shred_text_to_zettels(md_text, url, table, embed_model)
        else:
            raw_text = extract_text(path)
            if not raw_text or len(raw_text) < 100: continue
            _shred_text_to_zettels(raw_text, filename, table, embed_model)

        manifest[f_hash] = {"name": filename, "processed": datetime.now(timezone.utc).isoformat()}
        os.rename(path, os.path.join(RAW_ARCHIVE, filename))

    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=4)
    print("Compiling global structural graph cache...")
    router = LocalGraphRAGRouter(ZETTEL_DIR, DB_URI, embed_model=embed_model)

    # Clear the old cache to ensure newly ingested zettels are included in the cold-start build
    cache_file = Path(ZETTEL_DIR) / ".graph_cache.json"
    if cache_file.exists():
        os.remove(cache_file)

    router.rebuild_graph_from_vault()
    print("Ingestion Cycle Complete.")

class LocalGraphRAGRouter:
    """Compiles local flat text files into an active, searchable network topology graph."""

    def __init__(self, zettel_dir: str, db_uri: str, embed_model=None):
        self.zettel_dir = Path(zettel_dir)
        self.db = lancedb.connect(db_uri)
        self.embed_model = embed_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.graph = nx.DiGraph()
        self.vault: Dict[str, OKFZettel] = {}

    def rebuild_graph_from_vault(self):
        """Loads the entire topology from a single cached JSON block instantly, falling back to raw disk files if missing."""
        cache_path = self.zettel_dir / ".graph_cache.json"

        # 1. High-speed cache verification route
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                self.graph.clear()
                self.vault.clear()

                # Rehydrate network nodes along with their associated OKF metadata
                for node_id, node_metadata in cache_data.get("nodes", {}).items():
                    self.graph.add_node(node_id, **node_metadata)

                # Rehydrate network edge tracks
                for edge in cache_data.get("edges", []):
                    self.graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
                return  # Boot sequence completed instantly
            except Exception as e:
                print(f"      ! Cache read anomaly, fallback to scanning Markdown files: {e}")

        # 2. Cold-Start Fallback: Scrape flat-files if cache is missing or corrupt
        self.graph.clear()
        self.vault.clear()

        # Pass A: Hydrate structural nodes
        for md_file in self.zettel_dir.glob("*.md"):
            if md_file.name.startswith('.'): continue
            try:
                zettel = OKFZettel(md_file)
                z_id = zettel.metadata.get("id")
                if z_id:
                    self.vault[z_id] = zettel
                    self.graph.add_node(z_id, **zettel.metadata)
            except Exception:
                continue

        # Pass B: Inject inline relationships
        for z_id, zettel in self.vault.items():
            for relation, target_id in zettel.edges:
                if target_id in self.graph:
                    self.graph.add_edge(z_id, target_id, relation=relation)

        # 3. Save out compiled baseline state to prevent future cold-starts
        self.save_graph_cache()

    def save_graph_cache(self):
        """Serializes the live memory network graph state down to a single compact cache file."""
        cache_path = self.zettel_dir / ".graph_cache.json"
        cache_data = {"nodes": {}, "edges": []}

        for node, data in self.graph.nodes(data=True):
            cache_data["nodes"][node] = data
        for u, v, edge_data in self.graph.edges(data=True):
            cache_data["edges"].append({
                "source": u,
                "target": v,
                "relation": edge_data.get("relation", "references")
            })

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)

    def query(self, question: str, max_hops: int = 2, context: Optional[str] = None) -> str:
        """Executes a hybrid vector and topological multi-hop graph search query.

        If ``context`` is provided, it is used verbatim as the retrieved context
        (useful for external/long-context sources that cannot be typed inline);
        otherwise the context is assembled from the graph's vector + topology search.
        """
        # If external context was supplied, skip graph retrieval entirely and
        # synthesize directly over the caller-provided context.
        if context:
            sys_prompt = (
                "You are a Thinking Partner.\n"
                "Synthesize answers using the context below.\n"
                "Map hidden connections across cross-functional ideas.\n"
                "Cite insights using the node ID format: [[z_id]]."
            )
            usr_prompt = f"Question: {question}\n\nRELEVANT CONTEXT:\n{context}"
            return call_llm(usr_prompt, sys_prompt)

        # Instantly populates the graph topology via the cache map
        self.rebuild_graph_from_vault()
        table = self.db.open_table("notes")

        # 1. Search vector index to isolate seed nodes
        query_vector = self.embed_model.encode(question).tolist()
        vector_results = table.search(query_vector).limit(3).to_list()

        # Cross-reference vector matches against nodes present in our active graph
        seed_nodes = [r['id'] for r in vector_results if r['id'] != 'init' and r['id'] in self.graph]
        if not seed_nodes:
            return "No topological entry coordinates mapped for this specific query."

        # 2. Traverse paths across the network graph via ego networks
        combined_subgraph = nx.DiGraph()
        for seed in seed_nodes:
            ego = nx.ego_graph(self.graph, seed, radius=max_hops)
            combined_subgraph = nx.compose(combined_subgraph, ego)

        # 3. Metadata-First Protocol — assemble context to fit inside the hard
        #    token budget (CONTEXT_TOKEN_BUDGET), prioritising the vector-matched
        #    seed nodes and truncating bodies so output never gets starved.
        remaining_tokens = CONTEXT_TOKEN_BUDGET
        context_parts = []

        def add_part(text: str) -> bool:
            nonlocal remaining_tokens
            cost = _approx_tokens(text)
            if cost > remaining_tokens:
                return False
            context_parts.append(text)
            remaining_tokens -= cost
            return True

        if add_part("=== EXTRACTED GRAPH RELATIONAL PATHWAYS ==="):
            for u, v, data in combined_subgraph.edges(data=True):
                u_title = self.graph.nodes[u].get('title', u)
                v_title = self.graph.nodes[v].get('title', v)
                if not add_part(f"- Pathway: {u_title} -> [{data.get('relation', 'references')}] -> {v_title}"):
                    break

        add_part("\n=== CORE DATA PAYLOADS ===")

        # Seed nodes (direct vector hits) first, then the rest of the ego mesh
        ordered_nodes = seed_nodes + [n for n in combined_subgraph.nodes() if n not in seed_nodes]
        for node in ordered_nodes:
            # Dynamically read the file body from disk for the isolated subgraph nodes
            md_file = self.zettel_dir / f"{node}.md"
            if not md_file.exists():
                continue
            zettel = OKFZettel(md_file)
            header = (
                f"--- NOTE MATCH [[{node}]] ---\n"
                f"Title: {zettel.metadata.get('title')}\n"
                f"Summary: {zettel.metadata.get('summary')}\n"
            )
            header_cost = _approx_tokens(header) + _approx_tokens("Payload Body:\n")
            if header_cost > remaining_tokens:
                break
            # Spend whatever budget remains on the body, truncating on a word boundary
            body_chars = (remaining_tokens - header_cost) * _CHARS_PER_TOKEN
            body = zettel.content_body
            if len(body) > body_chars:
                body = body[:body_chars].rsplit(' ', 1)[0] + " ... [truncated to fit window]"
            if not add_part(f"{header}Payload Body:\n{body}\n"):
                break

        full_context = "\n".join(context_parts)

        # 4. Generate the query synthesis via the configured LLM provider
        sys_prompt = (
            "You are a Thinking Partner.\n"
            "Synthesize answers using the GraphRAG structural context below.\n"
            "Map hidden connections across cross-functional ideas.\n"
            "Cite insights using the node ID format: [[z_id]]."
        )

        usr_prompt = f"Question: {question}\n\nRELEVANT GRAPH CONTEXT VAULT:\n{full_context}"
        return call_llm(usr_prompt, sys_prompt)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        process_ingestion_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        # --file <context.txt> [--question "question"]
        # Synthesize directly over an external context file instead of typing it inline.
        args = sys.argv[2:]
        if not args:
            print("Usage: python3 knowledge_vault.py --file <context.txt> [--question \"...\"]")
            sys.exit(1)

        file_path = Path(args[0]).resolve()
        if not file_path.exists():
            print(f"\n      ! Context file not found: {file_path}")
            sys.exit(1)

        question = "Synthesize the key insights from the provided context and surface the most important cross-connections."
        if "--question" in args:
            idx = args.index("--question")
            if idx + 1 < len(args):
                question = args[idx + 1]

        with open(file_path, 'r', encoding='utf-8') as f:
            context = f.read()
        if not context.strip():
            print(f"\n      ! Context file is empty: {file_path}")
            sys.exit(1)

        print(f"\n--- GraphRAG External-Context Synthesis ---")
        print(f"Reading context from: {file_path}")
        router = LocalGraphRAGRouter(ZETTEL_DIR, DB_URI)
        answer = router.query(question, context=context)
        print(f"\n--- Synthesized Answer ---\n{answer}")
    else:
        print("\n--- Upgraded GraphRAG LLM-Wiki Query Loop ---")
        router = LocalGraphRAGRouter(ZETTEL_DIR, DB_URI)
        while True:
            query_str = input("\n[GraphRAG Search]: ")
            if query_str.lower() in ['exit', 'quit']: break
            print("Running structural multi-hop traversal across local files...")
            answer = router.query(query_str)
            print(f"\n--- Synthesized Answer ---\n{answer}")
