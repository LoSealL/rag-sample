import sys
from pathlib import Path
sys.path.insert(0, 'src')

from sec_rag.graph import CppAstExtractor, SymbolTable

st = SymbolTable()
extractor = CppAstExtractor(symbol_table=st)

source1 = '''
namespace rag {
float foo(float x) { return x * 2; }
} // namespace rag
'''
entities1, relations1 = extractor.extract(source1, "foo.cpp", "proj")
for e in entities1:
    st.register(e)
    print(f"Registered: {e.name} ({e.entity_type})")

source2 = '''
#include "foo.hpp"

float main() {
    return foo(42.0f);
}
'''
entities2, relations2 = extractor.extract(source2, "main.cpp", "proj")
for e in entities2:
    st.register(e)
    print(f"Registered: {e.name} ({e.entity_type})")

print("\nAll relations in main.cpp:")
for r in relations2:
    print(f"  {r.source.name} --{r.relation_type}--> {r.target.name}")

print("\nSymbol table lookup for 'foo':")
print(st.lookup("foo"))
