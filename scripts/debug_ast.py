import sys
from pathlib import Path
sys.path.insert(0, 'src')

import tree_sitter, tree_sitter_cpp

src = Path('.codebase/rag-test/include/rag/core/types.hpp').read_text()
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(bytes(src, 'utf-8'))
root = tree.root_node

def find_structs(node, depth=0):
    if node.type == 'struct_specifier':
        print(f'Struct at byte {node.start_byte}, line {node.start_point[0]+1}')
        print(f'  Raw bytes around start: {bytes(src, "utf-8")[node.start_byte:node.start_byte+20]}')
        print(f'  Source slice: "{src[node.start_byte:node.start_byte+20]}"')
        for c in node.children:
            if c.type == 'type_identifier':
                print(f'  type_identifier byte range: {c.start_byte}-{c.end_byte}')
                print(f'  type_identifier raw: {bytes(src, "utf-8")[c.start_byte:c.end_byte]}')
                print(f'  type_identifier source: "{src[c.start_byte:c.end_byte]}"')
    for c in node.children:
        find_structs(c, depth+1)

find_structs(root)
