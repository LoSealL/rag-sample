import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.llm_chunker import LLMChunker
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Create one LLM per thread
llm_local = LocalLLM()
chunker = LLMChunker(llm=llm_local, target_chunk_lines=80, overlap_lines=8, max_chunks=40)

reference = json.loads(Path('.codebase/rag-test/reference_labels.json').read_text())
codebase_path = Path('.codebase/rag-test')

def process_file(file_entry):
    file_path = codebase_path / file_entry["path"]
    if not file_path.exists():
        return None
    
    try:
        chunks = chunker.chunk_file(str(file_path))
    except Exception as e:
        return (file_entry["path"], f"ERROR: {e}")
    
    results = []
    for unit in file_entry["units"]:
        unit_name = unit["name"]
        unit_type = unit["type"]
        unit_start = unit["start_line"]
        unit_end = unit["end_line"]
        
        matching = [c for c in chunks if c.metadata.name == unit_name]
        containing = [c for c in chunks if c.metadata.line_start <= unit_start and c.metadata.line_end >= unit_end]
        
        if matching:
            is_exact = False
            for m in matching:
                type_match = (
                    m.metadata.chunk_type == unit_type
                    or (unit_type in ("class_template", "function_template", "struct_template") and m.metadata.chunk_type in ("class", "function", "struct"))
                    or (unit_type == "struct" and m.metadata.chunk_type == "class")
                )
                if type_match and m.metadata.line_start == unit_start and m.metadata.line_end == unit_end:
                    is_exact = True
                    break
            results.append({
                "file": file_entry["path"],
                "unit": unit_name,
                "type": unit_type,
                "detected": True,
                "exact": is_exact,
                "chunks": len(chunks)
            })
        elif containing:
            results.append({
                "file": file_entry["path"],
                "unit": unit_name,
                "type": unit_type,
                "detected": True,
                "exact": False,
                "chunks": len(chunks)
            })
        else:
            results.append({
                "file": file_entry["path"],
                "unit": unit_name,
                "type": unit_type,
                "detected": False,
                "exact": False,
                "chunks": len(chunks)
            })
    return results

print("Running LLMChunker evaluation in parallel...")
all_results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_file, f): f for f in reference["files"]}
    for future in as_completed(futures):
        result = future.result()
        if result:
            if isinstance(result, tuple):
                print(f"  {result[0]}: {result[1]}")
            else:
                all_results.extend(result)
                # Print summary for this file
                file_path = result[0]["file"]
                detected = sum(1 for r in result if r["detected"])
                exact = sum(1 for r in result if r["exact"])
                total = len(result)
                print(f"  {file_path}: {detected}/{total} detected, {exact}/{total} exact")

total = len(all_results)
detected = sum(1 for r in all_results if r["detected"])
exact = sum(1 for r in all_results if r["exact"])
missed = total - detected

print(f"\n{'='*60}")
print(f"LLMChunker (Improved Prompt) Results:")
print(f"  Total units: {total}")
print(f"  Detected:    {detected} ({detected/total*100:.1f}%)")
print(f"  Exact match: {exact} ({exact/total*100:.1f}%)")
print(f"  Missed:      {missed} ({missed/total*100:.1f}%)")
print(f"{'='*60}")

# Show missed units
missed_units = [r for r in all_results if not r["detected"]]
if missed_units:
    print("\nMissed units:")
    for r in missed_units[:10]:
        print(f"  - {r['file']}: {r['unit']} ({r['type']})")

# Show inexact units
inexact = [r for r in all_results if r["detected"] and not r["exact"]]
if inexact:
    print("\nInexact units (detected but wrong boundaries):")
    for r in inexact[:10]:
        print(f"  - {r['file']}: {r['unit']} ({r['type']})")
