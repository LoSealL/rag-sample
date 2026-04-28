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

llm = LocalLLM()
chunker = LLMChunker(llm=llm, target_chunk_lines=80, overlap_lines=8, max_chunks=40)

# Test types.hpp
file_path = '.codebase/rag-test/include/rag/core/types.hpp'
src = Path(file_path).read_text()

print("Testing types.hpp...")
chunks = chunker.chunk_source(src, file_path, "cpp")
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  {c.metadata.chunk_type}: {c.metadata.name} (lines {c.metadata.line_start}-{c.metadata.line_end})")

# Compare with reference
reference = json.loads(Path('.codebase/rag-test/reference_labels.json').read_text())
for f in reference['files']:
    if f['path'] == 'include/rag/core/types.hpp':
        print("\nReference units:")
        for u in f['units']:
            print(f"  {u['type']}: {u['name']} (lines {u['start_line']}-{u['end_line']})")
        
        print("\nComparison:")
        detected = 0
        exact = 0
        for u in f['units']:
            matches = [c for c in chunks if c.metadata.name == u['name']]
            if matches:
                detected += 1
                for m in matches:
                    if m.metadata.line_start == u['start_line'] and m.metadata.line_end == u['end_line']:
                        exact += 1
                        break
        print(f"  Detected: {detected}/{len(f['units'])}")
        print(f"  Exact: {exact}/{len(f['units'])}")
        break
