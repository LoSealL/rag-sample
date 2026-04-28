import re

lines = [
    "    /**",
    "     * @brief A point in 2D space.",
    "     */",
    "    struct Point {",
    "        float x;",
    "        float y;",
    "    };",
    "",
    "    float distance(Point a, Point b) {",
    "        return std::sqrt(a.x*a.x + a.y*a.y);",
    "    }",
]

keywords = {
    "class": r'^\s*class\s+',
    "class_template": r'^\s*template\s*<',
    "struct": r'^\s*struct\s+',
    "struct_template": r'^\s*template\s*<',
    "function": r'^\s*(?:[\w:<>,\s&*]+)\s+[A-Za-z_]\w*\s*\(',
    "function_template": r'^\s*template\s*<',
    "concept": r'^\s*template\s*<[^>]*>\s*concept\s+',
    "type_alias": r'^\s*using\s+',
    "namespace": r'^\s*namespace\s+',
}

for chunk_type, pattern in keywords.items():
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            print(f'{chunk_type}: matched line {i}: {repr(line)}')
