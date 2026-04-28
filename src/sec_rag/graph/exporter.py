"""Export the knowledge graph to external formats.

Supported formats:
- DOT (Graphviz) — quick textual visualisation with namespace clustering
- GEXF — import into Gephi, Cytoscape, etc.
- JSON — raw node-link data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .store import GraphStore


def export_to_dot(
    store: GraphStore,
    output_path: str,
    project_id: str | None = None,
    relation_types: list[str] | None = None,
    cluster_by_namespace: bool = True,
) -> None:
    """Export the graph (or a subgraph) to Graphviz DOT format.

    Args:
        store: GraphStore instance.
        output_path: Path to write the .dot file.
        project_id: If given, only export nodes from this project.
        relation_types: If given, only export edges of these types.
        cluster_by_namespace: If True, wrap namespace members in DOT
            subgraph clusters for cleaner layout.
    """
    lines = ["digraph KnowledgeGraph {"]
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=box, fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    # Collect nodes (including isolated ones) and edges
    nodes: set[str] = set()
    edges: list[tuple[str, str, dict[str, Any]]] = []

    # First add all nodes matching project filter
    for nid, d in store.nodes():
        if project_id and d.get("project_id") != project_id:
            continue
        nodes.add(nid)

    # Then collect edges
    for u, v, d in store.edges():
        if relation_types and d.get("relation_type") not in relation_types:
            continue
        if u not in nodes or v not in nodes:
            continue
        edges.append((u, v, d))

    # Node styling
    color_map = {
        "file": "lightblue",
        "namespace": "lightgreen",
        "class": "lightyellow",
        "struct": "lightyellow",
        "class_template": "lightyellow",
        "struct_template": "lightyellow",
        "function": "lightcoral",
        "function_template": "lightcoral",
        "type_alias": "lightpink",
        "concept": "plum",
    }

    # Group nodes by namespace for clustering
    namespace_members: dict[str, list[str]] = {}
    standalone_nodes: list[str] = []

    if cluster_by_namespace:
        # First pass: find namespace -> member mappings via CONTAINS edges
        for u, v, d in edges:
            if d.get("relation_type") == "CONTAINS":
                u_data = dict(store.nodes()).get(u, {})
                if u_data.get("entity_type") == "namespace":
                    namespace_members.setdefault(u, []).append(v)

        for nid in nodes:
            in_ns = any(nid in members for members in namespace_members.values())
            if not in_ns:
                standalone_nodes.append(nid)
    else:
        standalone_nodes = list(nodes)

    # Render namespace clusters
    for ns_id, members in namespace_members.items():
        ns_data = dict(store.nodes()).get(ns_id, {})
        ns_name = ns_data.get("name", ns_id)
        safe_ns = ns_name.replace('"', '\\"')
        cluster_id = f"cluster_{ns_id.replace(':', '_').replace('/', '_')}"
        lines.append(f'  subgraph {cluster_id} {{')
        lines.append(f'    label="{safe_ns}";')
        lines.append('    style=filled;')
        lines.append('    color=lightgrey;')
        lines.append('    fontsize=12;')
        for nid in members:
            data = dict(store.nodes()).get(nid, {})
            name = data.get("name", nid)
            etype = data.get("entity_type", "")
            color = color_map.get(etype, "white")
            safe_name = name.replace('"', '\\"')
            lines.append(f'    "{nid}" [label="{safe_name}", style=filled, fillcolor={color}];')
        lines.append('  }')

    # Render standalone nodes
    for nid in standalone_nodes:
        data = dict(store.nodes()).get(nid, {})
        name = data.get("name", nid)
        etype = data.get("entity_type", "")
        color = color_map.get(etype, "white")
        safe_name = name.replace('"', '\\"')
        lines.append(f'  "{nid}" [label="{safe_name}", style=filled, fillcolor={color}];')

    # Render edges
    for u, v, d in edges:
        rtype = d.get("relation_type", "")
        # Skip CONTAINS edges inside clusters (already implied by grouping)
        if cluster_by_namespace and rtype == "CONTAINS":
            continue
        lines.append(f'  "{u}" -> "{v}" [label="{rtype}"];')

    lines.append("}")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def export_to_gexf(
    store: GraphStore,
    output_path: str,
    project_id: str | None = None,
) -> None:
    """Export the graph to GEXF format (readable by Gephi)."""
    try:
        import networkx as nx
    except ImportError:
        raise RuntimeError("networkx is required for GEXF export")

    if project_id:
        all_nodes = [nid for nid, d in store.nodes() if d.get("project_id") == project_id]
        # Need to reconstruct subgraph
        g = nx.DiGraph()
        for nid, d in store.nodes():
            if nid in all_nodes:
                g.add_node(nid, **d)
        for u, v, d in store.edges():
            if u in all_nodes and v in all_nodes:
                g.add_edge(u, v, **d)
    else:
        g = nx.DiGraph()
        for nid, d in store.nodes():
            g.add_node(nid, **d)
        for u, v, d in store.edges():
            g.add_edge(u, v, **d)

    nx.write_gexf(g, output_path)


def export_to_json(store: GraphStore, output_path: str) -> None:
    """Export the raw node-link JSON."""
    data = {
        "nodes": [
            {"id": nid, **d}
            for nid, d in store.nodes()
        ],
        "edges": [
            {"source": u, "target": v, **d}
            for u, v, d in store.edges()
        ],
    }
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
