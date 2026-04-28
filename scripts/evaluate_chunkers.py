"""Evaluate chunker accuracy against reference_labels.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.chunkers.cpp_chunker import CppChunker
from sec_rag.chunkers.llm_chunker import LLMChunker


class MockLLM:
    """Mock LLM that returns empty response to trigger naive fallback."""
    def complete(self, prompt: str) -> str:
        return ""


class LocalLLM:
    """Real LLM client for localhost:8080."""
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


def load_reference():
    ref_path = Path(__file__).parent.parent / ".codebase" / "rag-test" / "reference_labels.json"
    with open(ref_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_type(chunk_type: str) -> str:
    """Normalize chunk type to reference type."""
    mapping = {
        "function": "function",
        "class": "class",
        "struct": "struct",
        "module": "namespace",
        "semantic": "other",
    }
    return mapping.get(chunk_type, chunk_type)


def evaluate_chunker(chunker, reference, codebase_path):
    """Evaluate a chunker against reference labels."""
    total_units = 0
    detected = 0
    exact_match = 0
    split_count = 0
    missed = 0
    
    details = []
    
    for file_entry in reference["files"]:
        file_path = codebase_path / file_entry["path"]
        if not file_path.exists():
            print(f"  SKIP (not found): {file_entry['path']}")
            continue
        
        print(f"  Processing: {file_entry['path']} ...", end=" ")
        try:
            chunks = chunker.chunk_file(str(file_path))
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print(f"OK ({len(chunks)} chunks)")
        
        for unit in file_entry["units"]:
            total_units += 1
            unit_name = unit["name"]
            unit_type = unit["type"]
            unit_start = unit["start_line"]
            unit_end = unit["end_line"]
            
            # Find chunks that match by name AND type AND line range (for disambiguation)
            matching = [
                c for c in chunks 
                if c.metadata.name == unit_name
                and (
                    c.metadata.chunk_type == unit_type
                    or (unit_type in ("class_template", "function_template", "struct_template") and c.metadata.chunk_type in ("class", "function", "struct"))
                    or (unit_type == "struct" and c.metadata.chunk_type == "class")
                )
                and c.metadata.line_start == unit_start
                and c.metadata.line_end == unit_end
            ]
            
            # If no exact match, try name + type (without line range)
            if not matching:
                matching = [
                    c for c in chunks 
                    if c.metadata.name == unit_name
                    and (
                        c.metadata.chunk_type == unit_type
                        or (unit_type in ("class_template", "function_template", "struct_template") and c.metadata.chunk_type in ("class", "function", "struct"))
                        or (unit_type == "struct" and c.metadata.chunk_type == "class")
                    )
                ]
            
            # If still no match, try name-only
            if not matching:
                matching = [c for c in chunks if c.metadata.name == unit_name]
            
            # Find chunks that contain this unit by line range
            containing = [
                c for c in chunks 
                if c.metadata.line_start <= unit_start and c.metadata.line_end >= unit_end
            ]
            
            if matching:
                detected += 1
                # Check exact match (lines)
                is_exact = False
                for m in matching:
                    if m.metadata.line_start == unit_start and m.metadata.line_end == unit_end:
                        is_exact = True
                        break
                
                if is_exact:
                    exact_match += 1
                elif len(matching) > 1 or len(containing) > 1:
                    split_count += 1
                    details.append({
                        "file": file_entry["path"],
                        "unit": unit_name,
                        "type": unit_type,
                        "issue": "split"
                    })
            elif containing:
                # Contained but name doesn't match — partial match
                detected += 1
                details.append({
                    "file": file_entry["path"],
                    "unit": unit_name,
                    "type": unit_type,
                    "issue": "contained_no_name"
                })
            else:
                missed += 1
                details.append({
                    "file": file_entry["path"],
                    "unit": unit_name,
                    "type": unit_type,
                    "issue": "missed"
                })
    
    return {
        "total": total_units,
        "detected": detected,
        "exact_match": exact_match,
        "split": split_count,
        "missed": missed,
        "detect_rate": detected / total_units * 100 if total_units else 0,
        "exact_rate": exact_match / total_units * 100 if total_units else 0,
        "split_rate": split_count / total_units * 100 if total_units else 0,
        "missed_rate": missed / total_units * 100 if total_units else 0,
        "details": details
    }


def print_result(name, result):
    print(f"\n--- {name} ---")
    print(f"  Total units: {result['total']}")
    print(f"  Detected:    {result['detected']:3d} ({result['detect_rate']:5.1f}%)")
    print(f"  Exact match: {result['exact_match']:3d} ({result['exact_rate']:5.1f}%)")
    print(f"  Split:       {result['split']:3d} ({result['split_rate']:5.1f}%)")
    print(f"  Missed:      {result['missed']:3d} ({result['missed_rate']:5.1f}%)")
    if result['details']:
        print(f"  Issues ({len(result['details'])}):")
        for d in result['details'][:10]:
            print(f"    - {d['file']}: {d['unit']} ({d['type']}) [{d['issue']}]")
        if len(result['details']) > 10:
            print(f"    ... and {len(result['details']) - 10} more")


def main():
    reference = load_reference()
    codebase_path = Path(__file__).parent.parent / ".codebase" / "rag-test"
    
    print("=" * 70)
    print("Chunker Evaluation against Reference Labels")
    print(f"Reference: {reference['metadata']}")
    print("=" * 70)
    
    # Test 1: CppChunker (AST)
    print("\n[1/3] CppChunker (AST-based)")
    cpp_chunker = CppChunker()
    result_cpp = evaluate_chunker(cpp_chunker, reference, codebase_path)
    print_result("CppChunker", result_cpp)
    
    # Test 2: LLMChunker with mock (naive fallback)
    print("\n[2/3] LLMChunker (Mock -> Naive Fallback)")
    llm_mock = MockLLM()
    llm_chunker_mock = LLMChunker(llm=llm_mock, target_chunk_lines=80, overlap_lines=8, max_chunks=40)
    result_mock = evaluate_chunker(llm_chunker_mock, reference, codebase_path)
    print_result("LLMChunker (naive)", result_mock)
    
    # Test 3: LLMChunker with real LLM
    print("\n[3/3] LLMChunker (Real LLM localhost:8080)")
    try:
        llm_real = LocalLLM()
        llm_chunker_real = LLMChunker(llm=llm_real, target_chunk_lines=80, overlap_lines=8, max_chunks=40)
        result_llm = evaluate_chunker(llm_chunker_real, reference, codebase_path)
        print_result("LLMChunker (improved prompt)", result_llm)
    except Exception as e:
        print(f"  Skipped: {e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
