import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.graph.ast_extractor import CppAstExtractor, SymbolTable
import tree_sitter, tree_sitter_cpp

st = SymbolTable()
extractor = CppAstExtractor(symbol_table=st)

source2 = '''
#include "foo.hpp"

float main() {
    return foo(42.0f);
}
'''

# Debug AST structure for main.cpp
lang = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser(lang)
tree = parser.parse(source2.encode("utf-8"))
root = tree.root_node

def show(node, depth=0):
    print('  ' * depth + f"{node.type} [{node.start_point[0]+1}:{node.start_point[1]}-{node.end_point[0]+1}:{node.end_point[1]}] = {repr(source2[node.start_byte:node.end_byte])}")
    for c in node.children:
        show(c, depth+1)

# Show only call_expression nodes
for node in extractor._iter_nodes(root):
    if node.type == "call_expression":
        print("call_expression found:")
        show(node, 0)
        print("  callee_name:", extractor._extract_callee_name(node, source2))
