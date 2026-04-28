import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.graph.ast_extractor import CppAstExtractor, SymbolTable

st = SymbolTable()
extractor = CppAstExtractor(symbol_table=st)

source1 = '''
namespace rag {
float foo(float x) { return x * 2; }
} // namespace rag
'''
entities1, relations1 = extractor.extract(source1, "foo.cpp", "proj")
print("Entities from foo.cpp:")
for e in entities1:
    print(f"  {e.entity_type}: {e.name}")

print(f"\nSymbol table after foo.cpp: {st._symbols}")

source2 = '''
#include "foo.hpp"

float main() {
    return foo(42.0f);
}
'''
entities2, relations2 = extractor.extract(source2, "main.cpp", "proj")
print("\nEntities from main.cpp:")
for e in entities2:
    print(f"  {e.entity_type}: {e.name}")

print(f"\nSymbol table after main.cpp: {st._symbols}")

print("\nCALLS relations in main.cpp:")
for r in relations2:
    if r.relation_type == "CALLS":
        print(f"  {r.source.name} calls {r.target.name}")
