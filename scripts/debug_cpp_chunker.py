import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.cpp_chunker import CppChunker

chunker = CppChunker()

# 手动调用 chunk_source 并打印中间结果
src = Path('.codebase/rag-test/include/rag/core/types.hpp').read_text()

import tree_sitter, tree_sitter_cpp
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, "utf-8"))
root = tree.root_node

print(f"Root has {len(root.children)} children")
for c in root.children:
    print(f"  {c.type}")

print("\nCalling _node_to_chunk on root children:")
for c in root.children:
    chunk = chunker._node_to_chunk(c, src, "test.cpp", "cpp")
    if chunk:
        print(f"  {c.type} -> {chunk.metadata.chunk_type}: {chunk.metadata.name}")
    else:
        print(f"  {c.type} -> None")

print("\nFull chunks output:")
chunks = chunker.chunk_source(src, ".codebase/rag-test/include/rag/core/types.hpp", "cpp")
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  {c.metadata.chunk_type}: {c.metadata.name}")
