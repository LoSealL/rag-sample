"""Demo: query knowledge graph for classes and functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.graph import GraphStore, GraphQuery

store = GraphStore(persist_path=".rag_index/knowledge_graph.json")
query = GraphQuery(store)

print("=== Classes in the graph ===")
for nid, data in store._g.nodes(data=True):
    if data.get("entity_type") in ("class", "class_template"):
        print(f"  {data['name']} ({data['file_path']}:{data['line_start']})")

print("\n=== Functions in embeddings.cpp ===")
for nid, data in store._g.nodes(data=True):
    if data.get("entity_type") == "function" and "embeddings.cpp" in data.get("file_path", ""):
        print(f"  {data['name']} (line {data['line_start']})")

print("\n=== Type aliases in types.hpp ===")
for nid, data in store._g.nodes(data=True):
    if data.get("entity_type") == "type_alias" and "types.hpp" in data.get("file_path", ""):
        print(f"  {data['name']} (line {data['line_start']})")

print("\n=== CONTAINS relations for namespace 'rag' ===")
for u, v, d in store._g.edges(data=True):
    if d.get("relation_type") == "CONTAINS":
        u_data = store._g.nodes[u]
        if u_data.get("name") == "rag":
            v_data = store._g.nodes[v]
            print(f"  {v_data.get('entity_type')}: {v_data.get('name')}")

print("\n=== INCLUDES relations ===")
for u, v, d in store._g.edges(data=True):
    if d.get("relation_type") == "INCLUDES":
        u_data = store._g.nodes[u]
        v_data = store._g.nodes[v]
        print(f"  {u_data.get('name')} -> {v_data.get('name')}")
