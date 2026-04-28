import sys
from pathlib import Path
sys.path.insert(0, 'src')

import tree_sitter, tree_sitter_cpp

src = Path('.codebase/rag-test/src/embeddings.cpp').read_text()
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, "utf-8"))
root = tree.root_node

def find_class_specifiers(node):
    if node.type == 'class_specifier':
        print(f'class_specifier at line {node.start_point[0]+1}')
        print(f'  children: {[c.type for c in node.children]}')
        print(f'  text: {repr(src[node.start_byte:node.end_byte])}')
    for c in node.children:
        find_class_specifiers(c)

find_class_specifiers(root)
