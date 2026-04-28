import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.llm_chunker import LLMChunker
import json

class LocalLLM:
    def __init__(self):
        import requests
        self._session = requests.Session()
        self._url = "http://localhost:8080/v1/chat/completions"
        self._model = "gemma-4-E2B-it-Q4_K_M.gguf"
    
    def complete(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a code analysis assistant. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        resp = self._session.post(self._url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

llm_local = LocalLLM()
chunker = LLMChunker(llm=llm_local, target_chunk_lines=80, overlap_lines=8, max_chunks=40)

reference = json.loads(Path('.codebase/rag-test/reference_labels.json').read_text())
codebase_path = Path('.codebase/rag-test')

for file_path_str in [
    "include/rag/core/concepts.hpp",
    "include/rag/embeddings/interface.hpp",
    "include/rag/store/flat.hpp",
    "src/embeddings.cpp",
    "src/store.cpp",
]:
    file_entry = next((f for f in reference["files"] if f["path"] == file_path_str), None)
    if not file_entry:
        continue
    file_path = codebase_path / file_path_str
    
    print(f"\n{'='*60}")
    print(f"{file_path_str}")
    print(f"{'='*60}")
    chunks = chunker.chunk_file(str(file_path))
    print(f"Chunks ({len(chunks)}):")
    for c in chunks:
        print(f"  {c.metadata.chunk_type}: {c.metadata.name} (lines {c.metadata.line_start}-{c.metadata.line_end})")
    
    print(f"Reference ({len(file_entry['units'])}):")
    for u in file_entry["units"]:
        print(f"  {u['type']}: {u['name']} (lines {u['start_line']}-{u['end_line']})")
    
    print("Differences:")
    for u in file_entry["units"]:
        matches = [c for c in chunks if c.metadata.name == u['name']]
        if matches:
            for m in matches:
                if m.metadata.line_start != u['start_line'] or m.metadata.line_end != u['end_line']:
                    print(f"  {u['name']}: ref {u['start_line']}-{u['end_line']}, got {m.metadata.line_start}-{m.metadata.line_end}")
        else:
            print(f"  {u['name']}: NOT FOUND")
