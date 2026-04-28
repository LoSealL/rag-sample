"""Phase 2 end-to-end demo: build graph with deep relations and query it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.graph import GraphBuilder, GraphQuery, GraphStore

codebase_path = Path(".codebase/rag-test")
store = GraphStore(persist_path=".rag_index/knowledge_graph.json")
builder = GraphBuilder(store=store)

# Rebuild graph with Phase 2 deep relations
cpp_files = list(codebase_path.rglob("*.hpp")) + list(codebase_path.rglob("*.cpp"))
print(f"Building graph from {len(cpp_files)} C++ files...")

for file_path in cpp_files:
    try:
        source = file_path.read_text(encoding="utf-8")
        rel_path = str(file_path.relative_to(codebase_path))
        builder.add_file(source, rel_path, "rag-test")
    except Exception as e:
        print(f"  Skip {file_path}: {e}")

builder.save()
stats = builder.stats()
print(f"\nGraph: {stats['nodes']} nodes, {stats['edges']} edges")

# Query deep relations
query = GraphQuery(store)

print("\n=== INHERITS_FROM relations ===")
for u, v, d in store._g.edges(data=True):
    if d.get("relation_type") == "INHERITS_FROM":
        u_data = store._g.nodes[u]
        v_data = store._g.nodes[v]
        print(f"  {u_data.get('name')} inherits from {v_data.get('name')}")

print("\n=== CALLS relations (sample) ===")
count = 0
for u, v, d in store._g.edges(data=True):
    if d.get("relation_type") == "CALLS" and count < 20:
        u_data = store._g.nodes[u]
        v_data = store._g.nodes[v]
        print(f"  {u_data.get('name')} calls {v_data.get('name')}")
        count += 1

print("\n=== REFERENCES relations (sample) ===")
count = 0
for u, v, d in store._g.edges(data=True):
    if d.get("relation_type") == "REFERENCES" and count < 15:
        u_data = store._g.nodes[u]
        v_data = store._g.nodes[v]
        print(f"  {u_data.get('name')} references {v_data.get('name')}")
        count += 1

print("\n=== Impact analysis for 'mean_vector' ===")
impact = query.impact_analysis("mean_vector")
print(f"  Direct callers: {len(impact['direct_callers'])}")
for e in impact['direct_callers'][:5]:
    print(f"    - {e.name} ({e.file_path}:{e.line_start})")
print(f"  Transitive deps: {len(impact['transitive_deps'])}")

print("\n=== Graph stats ===")
stats = store.stats()
for etype, count in sorted(stats.get("entities_by_type", {}).items()):
    print(f"  {etype}: {count}")

# Count relation types
rel_counts = {}
for _u, _v, d in store._g.edges(data=True):
    rt = d.get("relation_type", "UNKNOWN")
    rel_counts[rt] = rel_counts.get(rt, 0) + 1
print("\nRelation types:")
for rt, count in sorted(rel_counts.items()):
    print(f"  {rt}: {count}")
