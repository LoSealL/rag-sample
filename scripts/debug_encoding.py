import sys
from pathlib import Path
sys.path.insert(0, 'src')

src = Path('.codebase/rag-test/include/rag/core/types.hpp').read_text()

for i, ch in enumerate(src):
    if ord(ch) > 127:
        print(f'Non-ASCII at char {i}: U+{ord(ch):04X} ({ch})')
        # Show context
        start = max(0, i-20)
        end = min(len(src), i+20)
        print(f'  Context: {repr(src[start:end])}')
