"""Demo: build knowledge graph for .codebase/rag-test and query it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.graph import GraphBuilder, GraphQuery, GraphStore

# Build graph from .codebase/rag-test
codebase_path = Path(".codebase/rag-test")
store = GraphStore(persist_path=".rag_index/knowledge_graph.json")
builder = GraphBuilder(store=store)

# Find all C++ files
cpp_files = list(codebase_path.rglob("*.hpp")) + list(codebase_path.rglob("*.cpp"))
print(f"Found {len(cpp_files)} C++ files")

for file_path in cpp_files:
    try:
        source = file_path.read_text(encoding="utf-8")
        rel_path = str(file_path.relative_to(codebase_path))
        builder.add_file(source, rel_path, "rag-test")
    except Exception as e:
        print(f"  Skip {file_path}: {e}")

builder.save()
stats = builder.stats()
print(f"\nGraph built: {stats['nodes']} nodes, {stats['edges']} edges")
print("Entities by type:")
for etype, count in sorted(stats.get("entities_by_type", {}).items()):
    print(f"  {etype}: {count}")

# Query examples
query = GraphQuery(store)

print("\n--- Files that include types.hpp ---")
dependents = query.file_dependents("include/rag/core/types.hpp")
for d in dependents:
    print(f"  {d.name}")

print("\n--- Namespace rag contains ---")
from sec_rag.graph.models import EntityId
rag_id = EntityId.from_raw("rag-test", "include/rag/core/types.hpp", "namespace", "rag")
rag = store.get_entity(rag_id)
if rag:
    neighbors = store.get_neighbors(rag_id, relation_types=["CONTAINS"], direction="out")
    for entity, rel in neighbors:
        print(f"  {entity.entity_type}: {entity.name}")

print("\n--- Graph stats ---")
stats = store.stats()
print(f"Total nodes: {stats['nodes']}")
print(f"Total edges: {stats['edges']}")
