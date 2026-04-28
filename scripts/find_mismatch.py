import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.cpp_chunker import CppChunker
from sec_rag.chunkers.llm_chunker import LLMChunker
import json

ref_path = Path('.codebase/rag-test/reference_labels.json')
reference = json.loads(ref_path.read_text())
codebase_path = Path('.codebase/rag-test')

chunker = CppChunker()

for file_entry in reference["files"]:
    file_path = codebase_path / file_entry["path"]
    if not file_path.exists():
        continue
    
    chunks = chunker.chunk_file(str(file_path))
    
    for unit in file_entry["units"]:
        unit_name = unit["name"]
        unit_type = unit["type"]
        unit_start = unit["start_line"]
        unit_end = unit["end_line"]
        
        matching = [c for c in chunks if c.metadata.name == unit_name]
        exact = False
        for m in matching:
            type_match = (
                m.metadata.chunk_type == unit_type
                or (unit_type in ("class_template", "function_template") and m.metadata.chunk_type in ("class", "function"))
                or (unit_type == "struct" and m.metadata.chunk_type == "class")
            )
            if type_match and m.metadata.line_start == unit_start and m.metadata.line_end == unit_end:
                exact = True
                break
        
        if not exact:
            print(f"Not exact: {file_entry['path']} - {unit_name} ({unit_type}) lines {unit_start}-{unit_end}")
            for m in matching:
                print(f"  Match: {m.metadata.chunk_type} {m.metadata.name} lines {m.metadata.line_start}-{m.metadata.line_end}")
            # Also show chunks that contain this unit
            containing = [c for c in chunks if c.metadata.line_start <= unit_start and c.metadata.line_end >= unit_end]
            for c in containing:
                print(f"  Contain: {c.metadata.chunk_type} {c.metadata.name} lines {c.metadata.line_start}-{c.metadata.line_end}")
