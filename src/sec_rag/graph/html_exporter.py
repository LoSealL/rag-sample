"""Interactive HTML visualisation generator.

Produces a self-contained HTML file with a D3.js force-directed graph
that can be opened directly in a browser — no server required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


HTML_TEMPLATE = """\n<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Code Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{ margin: 0; font-family: Helvetica, Arial, sans-serif; background: #f5f5f5; }}
  #graph {{ width: 100vw; height: 100vh; }}
  .node {{ cursor: pointer; stroke: #fff; stroke-width: 1.5px; }}
  .node:hover {{ stroke: #000; stroke-width: 2px; }}
  .link {{ stroke: #999; stroke-opacity: 0.6; }}
  .label {{ font-size: 10px; pointer-events: none; text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff; }}
  #info {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.95); padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); max-width: 320px; font-size: 12px; }}
  #info h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
  #info table {{ width: 100%; border-collapse: collapse; }}
  #info td {{ padding: 2px 4px; }}
  #info td:first-child {{ color: #666; width: 80px; }}
  #controls {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.95); padding: 10px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  #controls input, #controls select {{ margin: 4px 0; width: 140px; }}
  #controls button {{ margin-top: 6px; width: 100%; }}
</style>
</head>
<body>
<div id="info">
  <h3>Code Knowledge Graph</h3>
  <table>
    <tr><td>Nodes:</td><td>{node_count}</td></tr>
    <tr><td>Edges:</td><td>{edge_count}</td></tr>
    <tr><td>Project:</td><td>{project_id}</td></tr>
  </table>
  <p style="margin:8px 0 0 0;color:#666;">Click a node to see details.<br>Drag to pan, scroll to zoom.</p>
  <div id="node-details" style="margin-top:10px;border-top:1px solid #ddd;padding-top:8px;display:none;">
    <strong id="detail-name"></strong>
    <p id="detail-meta" style="margin:4px 0;color:#444;"></p>
  </div>
</div>
<div id="controls">
  <div>Search: <input type="text" id="search" placeholder="node name..."></div>
  <div>Relation: <select id="relation-filter"><option value="">all</option>{relation_options}</select></div>
  <div>Type: <select id="type-filter"><option value="">all</option>{type_options}</select></div>
  <button onclick="resetZoom()">Reset View</button>
</div>
<svg id="graph"></svg>
<script>
const data = {json_data};

const width = window.innerWidth;
const height = window.innerHeight;
const colorMap = {color_map_json};

const svg = d3.select("#graph")
  .attr("width", width)
  .attr("height", height);

const g = svg.append("g");

const zoom = d3.zoom()
  .scaleExtent([0.1, 4])
  .on("zoom", (event) => g.attr("transform", event.transform));

svg.call(zoom);

function resetZoom() {{
  svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}}

let nodes = data.nodes.map(n => ({{ ...n, radius: n.entity_type === 'namespace' ? 18 : n.entity_type === 'class' ? 14 : 10 }}));
let links = data.edges.map(e => ({{ ...e, source: e.source, target: e.target }}));

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collide", d3.forceCollide().radius(d => d.radius + 5));

let link = g.append("g").attr("class", "links").selectAll("line").data(links).enter().append("line")
  .attr("class", "link")
  .attr("stroke-width", d => d.relation_type === 'INCLUDES' ? 1 : 2);

let node = g.append("g").attr("class", "nodes").selectAll("circle").data(nodes).enter().append("circle")
  .attr("class", "node")
  .attr("r", d => d.radius)
  .attr("fill", d => colorMap[d.entity_type] || "#ccc")
  .call(d3.drag()
    .on("start", (event, d) => {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
    .on("end", (event, d) => {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

let label = g.append("g").attr("class", "labels").selectAll("text").data(nodes).enter().append("text")
  .attr("class", "label")
  .attr("dy", d => d.radius + 12)
  .attr("text-anchor", "middle")
  .text(d => d.name);

node.on("click", (event, d) => {{
  document.getElementById("node-details").style.display = "block";
  document.getElementById("detail-name").textContent = d.name + " (" + d.entity_type + ")";
  document.getElementById("detail-meta").innerHTML =
    "File: " + (d.file_path || "-") + "<br>Line: " + d.line_start + "-" + d.line_end +
    (d.declaration ? "<br><code>" + d.declaration + "</code>" : "");
}});

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  node
    .attr("cx", d => d.x)
    .attr("cy", d => d.y);
  label
    .attr("x", d => d.x)
    .attr("y", d => d.y);
}});

// Search
d3.select("#search").on("input", function() {{
  const term = this.value.toLowerCase();
  node.style("opacity", d => d.name.toLowerCase().includes(term) || !term ? 1 : 0.1);
  label.style("opacity", d => d.name.toLowerCase().includes(term) || !term ? 1 : 0.1);
  link.style("opacity", d => (d.source.name.toLowerCase().includes(term) || d.target.name.toLowerCase().includes(term)) && term ? 0.8 : 0.05);
}});

// Relation filter
d3.select("#relation-filter").on("change", function() {{
  const type = this.value;
  link.style("display", d => !type || d.relation_type === type ? "block" : "none");
}});

// Type filter
d3.select("#type-filter").on("change", function() {{
  const type = this.value;
  node.style("display", d => !type || d.entity_type === type ? "block" : "none");
  label.style("display", d => !type || d.entity_type === type ? "block" : "none");
}});
</script>
</body>
</html>
"""


def export_to_html(
    store,
    output_path: str,
    project_id: str | None = None,
    relation_types: list[str] | None = None,
) -> None:
    """Export the graph to a self-contained interactive HTML file.

    The generated file uses D3.js (loaded from CDN) to render a
    force-directed graph with pan, zoom, search, and filtering.
    """
    # Collect nodes
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for nid, d in store.nodes():
        if project_id and d.get("project_id") != project_id:
            continue
        nodes.append({"id": nid, **d})
        node_ids.add(nid)

    # Collect edges
    edges: list[dict[str, Any]] = []
    for u, v, d in store.edges():
        if u not in node_ids or v not in node_ids:
            continue
        if relation_types and d.get("relation_type") not in relation_types:
            continue
        edges.append({"source": u, "target": v, **d})

    # Build relation type options
    rel_types = sorted(set(d.get("relation_type", "") for _u, _v, d in store.edges()))
    relation_options = "\n".join(f'<option value="{rt}">{rt}</option>' for rt in rel_types)

    # Build entity type options
    ent_types = sorted(set(d.get("entity_type", "") for _nid, d in store.nodes()))
    type_options = "\n".join(f'<option value="{et}">{et}</option>' for et in ent_types)

    color_map = {
        "file": "#a8d1ff",
        "namespace": "#b3e6b3",
        "class": "#fff4a3",
        "struct": "#fff4a3",
        "class_template": "#fff4a3",
        "struct_template": "#fff4a3",
        "function": "#ffb3b3",
        "function_template": "#ffb3b3",
        "type_alias": "#ffcce6",
        "concept": "#e6ccff",
        "template": "#ffd9b3",
    }

    import json as _json
    html = HTML_TEMPLATE.format(
        node_count=len(nodes),
        edge_count=len(edges),
        project_id=project_id or "all",
        relation_options=relation_options,
        type_options=type_options,
        json_data=_json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False),
        color_map_json=_json.dumps(color_map, ensure_ascii=False),
    )

    Path(output_path).write_text(html, encoding="utf-8")
