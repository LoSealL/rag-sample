"""Phase 4 end-to-end demo: incremental build, watch, and interactive HTML."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.graph import (
    GraphBuilder,
    GraphQuery,
    GraphStore,
    NetworkXBackend,
    FileHashIndex,
    export_to_html,
)

codebase_path = Path(".codebase/rag-test")
store = GraphStore(backend=NetworkXBackend(persist_path=".rag_index/knowledge_graph.json"))
hash_index = FileHashIndex(persist_path=".rag_index/file_hashes.json")
builder = GraphBuilder(store=store, hash_index=hash_index)

# Collect all C++ files
cpp_files = []
for ext in [".hpp", ".cpp"]:
    for file_path in codebase_path.rglob(f"*{ext}"):
        try:
            source = file_path.read_text(encoding="utf-8")
            rel_path = str(file_path.relative_to(codebase_path))
            cpp_files.append((rel_path, source))
        except Exception as e:
            print(f"Skip {file_path}: {e}")

# Demo 1: Initial build
print("=== Demo 1: Initial Build ===")
stats = builder.build_project(cpp_files, "rag-test")
print(f"Added: {stats['added']}, Updated: {stats['updated']}, Removed: {stats['removed']}, Skipped: {stats['skipped']}")
builder.save()

# Demo 2: Rebuild with no changes (should skip everything)
print("\n=== Demo 2: Rebuild (no changes) ===")
stats = builder.build_project(cpp_files, "rag-test")
print(f"Added: {stats['added']}, Updated: {stats['updated']}, Removed: {stats['removed']}, Skipped: {stats['skipped']}")
assert stats['skipped'] == len(cpp_files), "Should skip all unchanged files"

# Demo 3: Simulate a file change
print("\n=== Demo 3: Simulate File Change ===")
modified_files = cpp_files.copy()
# Modify the first file
modified_files[0] = (modified_files[0][0], modified_files[0][1] + "\n// modified\n")
stats = builder.build_project(modified_files, "rag-test")
print(f"Added: {stats['added']}, Updated: {stats['updated']}, Removed: {stats['removed']}, Skipped: {stats['skipped']}")
assert stats['updated'] == 1, "Should update exactly 1 file"
assert stats['skipped'] == len(cpp_files) - 1, "Should skip all others"
builder.save()

# Demo 4: Export interactive HTML
print("\n=== Demo 4: Interactive HTML Export ===")
output_html = ".rag_index/knowledge_graph.html"
export_to_html(store, output_html, project_id="rag-test")
print(f"Exported to {output_html}")
print(f"Open in browser: file://{Path(output_html).resolve()}")

# Demo 5: Show stats
print("\n=== Final Stats ===")
stats = store.stats()
print(f"Nodes: {stats['nodes']}")
print(f"Edges: {stats['edges']}")
print("Top entity types:")
for etype, count in sorted(stats.get("entities_by_type", {}).items(), key=lambda x: -x[1])[:8]:
    print(f"  {etype}: {count}")

print("\n=== CLI Commands ===")
print("Watch mode:")
print("  uv run python -m sec_rag.cli graph watch -p .codebase/rag-test --languages cpp")
print("\nExport interactive HTML:")
print("  uv run python -m sec_rag.cli graph export -f html -o graph.html")
print("\nExport DOT:")
print("  uv run python -m sec_rag.cli graph export -f dot -o graph.dot")
