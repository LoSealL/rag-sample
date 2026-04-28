import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.cpp_chunker import CppChunker

chunker = CppChunker()

# 调试 embeddings.cpp 的 template instantiation
file_path = '.codebase/rag-test/src/embeddings.cpp'
src = Path(file_path).read_text()

import tree_sitter, tree_sitter_cpp
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, "utf-8"))
root = tree.root_node

def find_all(node, depth=0):
    if node.type in ('class_specifier', 'struct_specifier', 'function_definition', 'alias_declaration', 'template_declaration', 'namespace_definition'):
        chunk = chunker._node_to_chunk(node, src, file_path, "cpp")
        if chunk:
            print(f'{node.type} at line {node.start_point[0]+1}: {chunk.metadata.chunk_type} / {chunk.metadata.name}')
    for c in node.children:
        find_all(c, depth+1)

find_all(root)
