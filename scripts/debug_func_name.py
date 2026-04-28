import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.chunkers.cpp_chunker import CppChunker

chunker = CppChunker()

# 调试 embeddings.cpp
file_path = '.codebase/rag-test/src/embeddings.cpp'
src = Path(file_path).read_text()

import tree_sitter, tree_sitter_cpp
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, "utf-8"))
root = tree.root_node

def find_functions(node, depth=0):
    if node.type == 'function_definition':
        print(f'Function at line {node.start_point[0]+1}')
        name = chunker._extract_name(node, src, "identifier")
        print(f'  Extracted name: "{name}"')
        # Show first few children types
        print(f'  Children types: {[c.type for c in node.children[:5]]}')
        for c in node.children:
            if c.type == 'function_declarator':
                print(f'    function_declarator children: {[cc.type for cc in c.children]}')
                for cc in c.children:
                    if cc.type == 'identifier':
                        print(f'      identifier: "{chunker._text_from_node(src, cc)}"')
    for c in node.children:
        find_functions(c, depth+1)

find_functions(root)
