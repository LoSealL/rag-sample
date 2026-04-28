import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.cpp_chunker import CppChunker

chunker = CppChunker()

# 调试 embeddings.cpp
file_path = '.codebase/rag-test/src/embeddings.cpp'
src = Path(file_path).read_text()

chunks = chunker.chunk_source(src, file_path, "cpp")
print(f'Total chunks: {len(chunks)}')
for c in chunks:
    print(f'  {c.metadata.chunk_type}: {c.metadata.name} (lines {c.metadata.line_start}-{c.metadata.line_end})')

# 查看 AST
import tree_sitter, tree_sitter_cpp
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, "utf-8"))
root = tree.root_node

def show_types(node, depth=0, max_depth=2):
    if depth > max_depth:
        return
    print('  ' * depth + node.type)
    for c in node.children:
        show_types(c, depth+1, max_depth)

print("\nAST (top 2 levels):")
show_types(root, 0, 2)
