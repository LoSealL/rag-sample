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

# Select representative files
selected_files = [
    "include/rag/core/types.hpp",
    "include/rag/core/concepts.hpp",
    "include/rag/embeddings/interface.hpp",
    "include/rag/store/flat.hpp",
    "src/embeddings.cpp",
    "src/store.cpp",
]

total = 0
detected = 0
exact = 0

for file_path_str in selected_files:
    file_entry = next((f for f in reference["files"] if f["path"] == file_path_str), None)
    if not file_entry:
        continue
    
    file_path = codebase_path / file_path_str
    print(f"Processing {file_path_str} ...")
    chunks = chunker.chunk_file(str(file_path))
    print(f"  Got {len(chunks)} chunks")
    
    for unit in file_entry["units"]:
        total += 1
        unit_name = unit["name"]
        unit_type = unit["type"]
        unit_start = unit["start_line"]
        unit_end = unit["end_line"]
        
        matching = [c for c in chunks if c.metadata.name == unit_name]
        if matching:
            detected += 1
            for m in matching:
                type_match = (
                    m.metadata.chunk_type == unit_type
                    or (unit_type in ("class_template", "function_template", "struct_template") and m.metadata.chunk_type in ("class", "function", "struct"))
                    or (unit_type == "struct" and m.metadata.chunk_type == "class")
                )
                if type_match and m.metadata.line_start == unit_start and m.metadata.line_end == unit_end:
                    exact += 1
                    break

print(f"\n{'='*60}")
print(f"LLMChunker (Improved Prompt) — Representative Sample:")
print(f"  Files evaluated: {len(selected_files)}")
print(f"  Total units: {total}")
print(f"  Detected:    {detected} ({detected/total*100:.1f}%)")
print(f"  Exact match: {exact} ({exact/total*100:.1f}%)")
print(f"{'='*60}")
