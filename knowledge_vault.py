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

# 1. Silence all Hugging Face Hub connection warnings and logger noise
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIGURATION ---
PROVIDER = "LMSTUDIO"
MODEL_NAME = "google/gemma-4-e4b"
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"

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
            self.metadata = {"id": self.file_path.stem, "title": self.file_path.stem, "type": "concept", "tags": [], "summary": ""}

        self.edges = self.LINK_PATTERN.findall(self.content_body)

    @classmethod
    def write_to_disk(cls, directory: Path, z_id: str, title: str, z_type: str,
                      summary: str, tags: List[str], body: str) -> 'OKFZettel':
        """Enforces clean schema verification before writing files to local disk storage."""
        file_path = directory / f"{z_id}.md"
        metadata = {
            "okf_version": "1.0",
            "id": z_id,
            "title": title,
            "type": z_type,
            "tags": tags,
            "summary": summary,
            "last_modified": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }
        yaml_header = yaml.dump(metadata, sort_keys=False, allow_unicode=True)
        full_markdown = f"---\n{yaml_header}---\n\n# {title}\n\n{body.strip()}\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
        return cls(file_path)

# --- UTILITIES & LLM INTERACTION ---

def call_llm(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Sends requests to local inference endpoint, returning raw generated text strings safely."""
    url = LMSTUDIO_URL if PROVIDER == "LMSTUDIO" else OLLAMA_URL
    headers = {"Content-Type": "application/json"}

    if PROVIDER == "LMSTUDIO":
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    else:
        payload = {"model": MODEL_NAME, "prompt": prompt, "system": system_prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if response.status_code != 200:
            print(f"      ! HTTP Error: Server responded with status {response.status_code}")
            return None

        res_obj = response.json()
        raw_output = res_obj['choices'][0]['message']['content'] if PROVIDER == "LMSTUDIO" else res_obj.get('response', '')

        if not raw_output or not raw_output.strip():
            print(f"\n[DEBUG] Empty response generated. Raw Server JSON Payload:\n{res_obj}\n")
            return None

        return raw_output
    except Exception as e:
        print(f"      ! Connection Failure: {e}")
        return None

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

def process_ingestion_pipeline():
    """Builds atomic OKF nodes using plain text structural tagging rather than fragile JSON schemas."""
    print("Initializing Database Connections...")
    db = lancedb.connect(DB_URI)
    try:
        table = db.open_table("notes")
    except Exception:
        table = db.create_table("notes", data=[{"vector": [0.0]*384, "id": "init", "text": "root"}])

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
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
        raw_text = extract_text(path)
        if not raw_text or len(raw_text) < 100: continue

        # Keep chunk size tight for local model context safety
        chunk_size, overlap = 3000, 400
        chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size - overlap)]
        print(f"    > Shredding into {len(chunks)} chunks...")

        for idx, chunk in enumerate(chunks):
            print(f"      Processing chunk {idx + 1}/{len(chunks)}...")

            shred_sys = (
                "You are an Advanced Knowledge Architect. Deconstruct the incoming text into atomic, self-contained zettel concepts.\n"
                "For each distinct concept identified, output it using exactly the following plain-text structural layout:\n\n"
                "<<<START_ZETTEL>>>\n"
                "TITLE: Title of the concept\n"
                "TYPE: concept\n"
                "TAGS: tag1, tag2\n"
                "SUMMARY: Short structural overview under 50 words.\n"
                "RELATED: Key Topic A, Key Topic B\n"
                "BODY:\n"
                "Deep-dive analytical description, minimum 200 words. Explain Mechanics and Implications.\n"
                "<<<END_ZETTEL>>>"
            )

            raw_output = call_llm(f"TARGET CONTENT TO TRANSLATE:\n{chunk}", shred_sys)
            if not raw_output:
                print(f"      ! Chunk {idx + 1} failed: The local model returned a blank string. Check your LM Studio Context Slider!")
                continue

            # Parse out zettel text blocks cleanly using text tokens
            zettel_blocks = re.findall(r'<<<START_ZETTEL>>>(.*?)<<<END_ZETTEL>>>', raw_output, re.DOTALL)

            if not zettel_blocks:
                print(f"      ! Chunk {idx + 1} skipped: Could not locate text bounds (<<<START_ZETTEL>>>).")
                continue

            for block in zettel_blocks:
                # Direct line-by-line field scrapers
                def get_field(field_name: str) -> str:
                    m = re.search(rf'^\s*{field_name}\s*:\s*(.*)$', block, re.MULTILINE | re.IGNORECASE)
                    return m.group(1).strip() if m else ""

                title_val = get_field("TITLE")
                type_val = get_field("TYPE") or "concept"
                summary_val = get_field("SUMMARY")
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
                    body=final_body
                )

                # Track position updates inside LanceDB index
                embedding = embed_model.encode(body_val).tolist()
                table.add([{"vector": embedding, "id": z_id, "text": f"{title_val}: {summary_val}"}])
                print(f"      + Successfully Ingested OKF Zettel: {z_id} ({title_val[:40]}...)")

        manifest[f_hash] = {"name": filename, "processed": datetime.now(timezone.utc).isoformat()}
        os.rename(path, os.path.join(RAW_ARCHIVE, filename))

    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=4)
    print("Compiling global structural graph cache...")
    router = LocalGraphRAGRouter(ZETTEL_DIR, DB_URI)

    # Clear the old cache to ensure newly ingested zettels are included in the cold-start build
    cache_file = Path(ZETTEL_DIR) / ".graph_cache.json"
    if cache_file.exists():
        os.remove(cache_file)

    router.rebuild_graph_from_vault()
    print("Ingestion Cycle Complete.")

class LocalGraphRAGRouter:
    """Compiles local flat text files into an active, searchable network topology graph."""

    def __init__(self, zettel_dir: str, db_uri: str):
        self.zettel_dir = Path(zettel_dir)
        self.db = lancedb.connect(db_uri)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
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

    def query(self, question: str, max_hops: int = 2) -> str:
        """Executes a hybrid vector and topological multi-hop graph search query."""
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

        # 3. Apply the Metadata-First Protocol to optimize context packaging
        context_parts = []
        context_parts.append("=== EXTRACTED GRAPH RELATIONAL PATHWAYS ===")
        for u, v, data in combined_subgraph.edges(data=True):
            u_title = self.graph.nodes[u].get('title', u)
            v_title = self.graph.nodes[v].get('title', v)
            context_parts.append(f"- Pathway: {u_title} -> [{data.get('relation', 'references')}] -> {v_title}")

        context_parts.append("\n=== CORE DATA PAYLOADS ===")
        for node in combined_subgraph.nodes():
            # Dynamically read the file body from disk for the isolated subgraph nodes
            md_file = self.zettel_dir / f"{node}.md"
            if md_file.exists():
                zettel = OKFZettel(md_file)
                context_parts.append(
                    f"--- NOTE MATCH [[{node}]] ---\n"
                    f"Title: {zettel.metadata.get('title')}\n"
                    f"Summary: {zettel.metadata.get('summary')}\n"
                    f"Payload Body:\n{zettel.content_body}\n"
                )

        full_context = "\n".join(context_parts)

        # 4. Generate the query synthesis via the local LLM
        sys_prompt = (
            "You are a Thinking Partner. Synthesize a comprehensive answer using the provided GraphRAG structural context.\n"
            "Map hidden connections across cross-functional ideas by utilizing the structural relation links.\n"
            "Always cite relevant insights by using the explicit node ID format: [[z_id]]."
        )

        usr_prompt = f"Question: {question}\n\nRELEVANT GRAPH CONTEXT VAULT:\n{full_context}"
        return call_llm(usr_prompt, sys_prompt)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        process_ingestion_pipeline()
    else:
        print("\n--- Upgraded GraphRAG LLM-Wiki Query Loop ---")
        router = LocalGraphRAGRouter(ZETTEL_DIR, DB_URI)
        while True:
            query_str = input("\n[GraphRAG Search]: ")
            if query_str.lower() in ['exit', 'quit']: break
            print("Running structural multi-hop traversal across local files...")
            answer = router.query(query_str)
            print(f"\n--- Synthesized Answer ---\n{answer}")
