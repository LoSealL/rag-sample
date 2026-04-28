"""Phase 3 end-to-end demo: incremental updates and namespace-clustered export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_rag.graph import GraphBuilder, GraphQuery, GraphStore, NetworkXBackend, export_to_dot

# Setup
codebase_path = Path(".codebase/rag-test")
store = GraphStore(backend=NetworkXBackend(persist_path=".rag_index/knowledge_graph.json"))
builder = GraphBuilder(store=store)

# Initial build
cpp_files = list(codebase_path.rglob("*.hpp")) + list(codebase_path.rglob("*.cpp"))
print(f"Initial build from {len(cpp_files)} files...")
for file_path in cpp_files:
    source = file_path.read_text(encoding="utf-8")
    rel_path = str(file_path.relative_to(codebase_path))
    builder.add_file(source, rel_path, "rag-test")

builder.save()
stats = builder.stats()
print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")

# Demo 1: Incremental update
print("\n=== Demo: Incremental Update ===")
print("Updating src/chunking.cpp with a new function...")

new_chunking = '''
#include "rag/chunking/fixed.hpp"
#include "rag/chunking/sliding.hpp"

#include <cctype>
#include <string>
#include <string_view>

namespace rag {

std::size_t approximate_word_count(std::string_view text) {
    // NEW: improved word counting
    std::size_t count = 0;
    bool in_word = false;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            if (!in_word) {
                ++count;
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    return count;
}

std::string sanitise_text(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (char ch : raw) {
        if (std::isprint(static_cast<unsigned char>(ch))) {
            out.push_back(ch);
        } else {
            out.push_back(' ');
        }
    }
    return out;
}

std::size_t estimate_token_count(std::string_view text) {
    return approximate_word_count(text) * 4 / 3 + 1;
}

// NEW FUNCTION
float compute_ratio(std::size_t a, std::size_t b) {
    return b == 0 ? 0.0f : static_cast<float>(a) / static_cast<float>(b);
}

} // namespace rag
'''

builder.update_file(new_chunking, "src/chunking.cpp", "rag-test")
builder.save()

query = GraphQuery(store)
new_func = query._find_entity("rag::compute_ratio", "function", "rag-test")
if new_func:
    print(f"  New function detected: {new_func.name} ({new_func.file_path}:{new_func.line_start})")
else:
    print("  New function not found (may have different name)")

# Check old function still exists
old_func = query._find_entity("rag::sanitise_text", "function", "rag-test")
if old_func:
    print(f"  Old function preserved: {old_func.name}")

# Demo 2: Namespace-clustered DOT export
print("\n=== Demo: Namespace-Clustered DOT Export ===")
export_to_dot(
    store,
    ".rag_index/knowledge_graph.dot",
    cluster_by_namespace=True,
    relation_types=["CALLS", "REFERENCES", "INHERITS_FROM", "INCLUDES"],
)
print("  Exported to .rag_index/knowledge_graph.dot")
print("  Render with: dot -Tpng .rag_index/knowledge_graph.dot -o graph.png")

# Demo 3: Stats
print("\n=== Final Stats ===")
stats = store.stats()
print(f"  Nodes: {stats['nodes']}")
print(f"  Edges: {stats['edges']}")
print("  Top entity types:")
for etype, count in sorted(stats.get("entities_by_type", {}).items(), key=lambda x: -x[1])[:8]:
    print(f"    {etype}: {count}")

# Relation breakdown
rel_counts = {}
for _u, _v, d in store.edges():
    rt = d.get("relation_type", "UNKNOWN")
    rel_counts[rt] = rel_counts.get(rt, 0) + 1
print("  Relation types:")
for rt, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
    print(f"    {rt}: {count}")
