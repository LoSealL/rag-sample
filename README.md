# sec-rag

本地 RAG 系统，支持代码和文档的多项目索引、代码脱敏、安全扫描，以及**代码知识图谱**。

索引 Python/C/C++/SystemC/Verilog 代码库以及 Markdown/Word 文档，跨多个项目进行统一管理。通过自然语言查询，只将最相关的代码块发送给 LLM，大幅降低 token 消耗。同时自动检测并拦截敏感信息，并构建了可查询的**代码知识图谱**来支持深度代码理解。

## 功能特性

### 多项目索引
- 一次索引多个项目目录：`--path proj1 --path proj2`
- 每个项目拥有独立摘要，作为检索入口
- 两阶段检索：项目摘要 → 函数/模块级别
- 按项目、语言、文档类型进行元数据过滤

### 代码脱敏与安全
- **C/C++/SystemC 自动脱敏**：
  - 仅提取函数声明（移除实现体）
  - 通过 LLM 生成行为摘要（输入、输出、状态转换）
  - 生成 9 行脱敏文档（元数据 + 声明 + 摘要）
- **敏感信息检测**（脱敏后二次检查）：
  - 拦截密码、邮箱、IP、电话、带凭证的 URL
  - 硬失败策略：检测到敏感信息时立即抛出 `SensitiveInfoError`
  - 适用于所有已索引文档，包括项目摘要

### 隐私模式（新功能）
- **高密级文件标记**：用户可以指定某些文件为高密级
- **四种匹配规则**：
  - 精确路径匹配：`src/secrets/api_keys.h`
  - 目录匹配：`src/crypto/`（该目录下所有文件）
  - 正则匹配：`.*_secret\.py$`
  - 扩展名匹配：`.pem`、`.key`
- **索引时生成摘要**：通过 LLM 生成隐私安全摘要，描述功能但不暴露实现
- **查询时自动替换**：隐私文件的原文绝不会进入 LLM Prompt，自动替换为摘要
- **醒目日志**：涉及隐私文件时打印黄色警告日志（`[PRIVACY ALERT]`）

### 语言支持
| 语言 | 分块方式 | 图谱支持 |
|------|---------|---------|
| Python | AST（函数、类） | ✅ 实体 + 调用链 |
| C/C++ | tree-sitter AST → 脱敏 → LLM 摘要 | ✅ 实体 + 深层关系 |
| SystemC/TLM | 自动检测（`#include <systemc.h>`、`SC_MODULE`），C++ 解析器 + 特殊标记 | ✅ |
| Verilog/SystemVerilog | 正则表达式（模块/端口/always 块） | ❌ |
| 文档 | Markdown、Word（`unstructured.io`） | ❌ |

### 代码知识图谱（新功能）
- **AST 实体提取**：函数、类、结构体、类型别名、命名空间、模板、概念
- **深层关系**：
  - `CONTAINS` — 文件包含实体
  - `INCLUDES` — 头文件引用
  - `CALLS` — 函数调用链
  - `REFERENCES` — 类型引用（参数、成员变量）
  - `INHERITS_FROM` — 类继承关系
- **跨文件解析**：项目级符号表自动解析跨文件引用
- **增量更新**：基于 SHA-256 文件哈希，只处理变更文件
- **Watch 模式**：轮询监控目录，文件变更时自动更新图谱
- **交互式可视化**：D3.js 力导向图（搜索、过滤、缩放）
- **混合检索**：向量语义检索 + 图谱关系扩展

### 本地模型灵活性
- **LLM**：兼容 OpenAI 或 Anthropic API（自动检测）
  - 默认：`http://localhost:8000`（qwen3.5-8b-instruct-awq）
- **Embeddings**：兼容任意 Embedding API
  - 默认：`http://localhost:8000`（qwen3-vl-embedding-8b）
- 通过 `--api-base-url`、`--llm-provider`、`--llm-model` 覆盖

---

## 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repo-url>
cd sec-rag

# 安装依赖
pip install -e .

# 或推荐方式（使用 uv）
uv pip install -e .
```

系统依赖（Ubuntu/Debian）：
```bash
sudo apt install libmagic-dev poppler-utils
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env，配置本地模型网关
```

`.env` 示例：
```bash
# 本地模型网关
LOCAL_LLM_BASE_URL=http://localhost:8000
LLM_PROVIDER=auto
LLM_MODEL=Qwen3.5-8B
EMBEDDING_MODEL=Qwen3-VL-Embedding-8B

# 索引存储位置
RAG_INDEX_DIR=.rag_index
```

### 3. 索引项目（自动构建知识图谱）

```bash
# 单项目索引
sec-rag-index --path ./src --languages python,c,cpp

# 多项目索引（每个项目独立摘要）
sec-rag-index --path ./frontend --path ./backend --languages python,cpp,verilog
```

索引完成后，知识图谱自动保存在 `.rag_index/knowledge_graph.json`。

---

## 分步使用指南

### 步骤 1：查看知识图谱统计

```bash
sec-rag graph stats
```

输出示例：
```
=== Graph Statistics ===
Nodes: 260
Edges: 485

Entity types:
  function: 115
  file: 63
  namespace: 24
  template: 19
  class: 14
  type_alias: 13
  struct: 7
  concept: 5

Top relation types:
  CONTAINS: 261
  INCLUDES: 160
  REFERENCES: 57
  CALLS: 4
  INHERITS_FROM: 2
```

### 步骤 2（可选）：配置隐私模式

创建 `.privacy_config.json` 文件标记高密级文件：

```json
{
  "exact_paths": ["src/secrets/api_keys.h", "config/production.yml"],
  "directories": ["src/crypto/", "internal/"],
  "patterns": [".*_secret\\.py$", ".*\\.token$"],
  "extensions": [".pem", ".p12"],
  "enabled": true
}
```

匹配规则说明：
- `exact_paths`：精确文件路径（支持后缀匹配）
- `directories`：目录下所有文件（递归）
- `patterns`：正则表达式匹配文件路径
- `extensions`：按扩展名匹配

索引时指定配置文件：
```bash
sec-rag-index --path ./src --languages cpp --privacy-config .privacy_config.json
```

系统会自动检测项目根目录下的 `.privacy_config.json`，无需手动指定。

### 步骤 3：查询实体

**查找函数调用链（谁调用了我）**：
```bash
sec-rag graph callers -n l2_norm
```

**查找函数被谁调用（我调用了谁）**：
```bash
sec-rag graph callees -n process_data
```

**查看类继承层次**：
```bash
sec-rag graph hierarchy -n Document
```

**查看文件依赖**：
```bash
# 查看文件引用了哪些头文件
sec-rag graph deps -f src/main.cpp

# 查看哪些文件引用了该头文件
sec-rag graph dependents -f include/utils.h
```

### 步骤 4：影响范围分析

分析修改某个实体会影响哪些代码：

```bash
sec-rag graph impact -n Document
```

输出示例：
```
Impact analysis for Document (file: src/document.h)

Callers (upstream):
  - index_document (file: src/indexer.cpp)
  - render_document (file: src/renderer.cpp)

Downstream (calls from Document):
  - validate (file: src/validator.cpp)
```

### 步骤 5：导出可视化

**交互式 HTML（推荐）**：
```bash
sec-rag graph export -f html -o graph.html
# 用浏览器打开 graph.html
```

特性：
- 力导向图布局（D3.js）
- 搜索框：按名称搜索实体
- 关系过滤：选择显示哪些关系类型
- 类型过滤：选择显示哪些实体类型
- 缩放/平移/拖拽
- 双击实体高亮其关系

**导出为 DOT（Graphviz）**：
```bash
sec-rag graph export -f dot -o graph.dot
# dot -Tpng graph.dot -o graph.png
```

**导出为 GEXF（Gephi）**：
```bash
sec-rag graph export -f gexf -o graph.gexf
```

**导出为 JSON**：
```bash
sec-rag graph export -f json -o graph.json
```

### 步骤 6：Watch 模式（自动更新）

持续监控目录，文件变更时自动增量更新知识图谱：

```bash
sec-rag graph watch -p ./src --languages cpp --interval 5.0
```

参数：
- `-p, --path`：监控的目录路径
- `--languages`：逗号分隔的语言列表（`python`, `c`, `cpp`）
- `--interval`：轮询间隔（秒），默认 5.0

按 `Ctrl+C` 停止监控。

### 步骤 7：重新索引（增量更新）

再次运行索引命令时，系统会自动：
1. 计算每个文件的 SHA-256 哈希
2. 对比上一次索引的哈希值
3. 只处理**新增、修改、删除**的文件
4. 未变更文件自动跳过

```bash
# 第一次：处理 23 个文件
sec-rag-index --path ./src --languages cpp
# → Added: 23, Updated: 0, Removed: 0, Skipped: 0

# 第二次：无变更
sec-rag-index --path ./src --languages cpp
# → Added: 0, Updated: 0, Removed: 0, Skipped: 23

# 修改一个文件后
sec-rag-index --path ./src --languages cpp
# → Added: 0, Updated: 1, Removed: 0, Skipped: 22
```

---

## CLI 完整命令参考

### `sec-rag-index` — 索引项目

```bash
sec-rag-index \
  --path ./project-a \
  --path ./project-b \
  --languages python,c,cpp,verilog,doc \
  --batch-size 10 \
  --api-base-url http://localhost:8000 \
  --llm-provider auto \
  --llm-model Qwen3.5-8B \
  --verbose
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--path` | （必需） | 项目目录，可多次指定 |
| `--languages` | `python,c,doc` | `python`, `c`, `cpp`, `systemc`, `verilog`, `systemverilog`, `doc` |
| `--batch-size` | `10` | 每批处理的文件数 |
| `--privacy-config` | （自动检测） | 隐私配置文件路径（`.privacy_config.json`） |
| `--api-base-url` | `http://localhost:8000` | 本地模型网关 URL |
| `--llm-provider` | `auto` | `auto`, `openai`, `anthropic` |
| `--llm-model` | `Qwen3.5-8B` | LLM 模型名 |
| `--verbose` | （关闭） | 调试日志 |

### `sec-rag-query` — 自然语言查询

```bash
sec-rag-query "DMA 控制器是如何初始化的？"
```

流程：
1. Embedding 查询文本
2. 从 `project_index` 检索 Top-3 最相关项目
3. 在每个项目中检索 Top-5 代码块 + Top-3 文档块
4. 按相似度排序
5. 组装最终 Prompt（Top-10 唯一块）
6. 发送给 LLM

### `sec-rag graph` — 知识图谱命令

```
sec-rag graph [命令] [选项]
```

| 子命令 | 说明 | 示例 |
|--------|------|------|
| `stats` | 查看图谱统计 | `sec-rag graph stats` |
| `callers` | 查询调用者（谁调用了我） | `sec-rag graph callers -n foo` |
| `callees` | 查询被调用者（我调用了谁） | `sec-rag graph callees -n foo` |
| `hierarchy` | 查看类继承层次 | `sec-rag graph hierarchy -n MyClass` |
| `deps` | 查看文件包含的头文件 | `sec-rag graph deps -f main.cpp` |
| `dependents` | 查看哪些文件包含该头文件 | `sec-rag graph dependents -f utils.h` |
| `impact` | 影响范围分析 | `sec-rag graph impact -n MyClass` |
| `export` | 导出图谱 | `sec-rag graph export -f html -o out.html` |
| `watch` | Watch 模式自动更新 | `sec-rag graph watch -p ./src` |

### `sec-rag-stats` — 查看索引统计

```bash
sec-rag-stats
```

---

## 架构说明

### 整体架构

```
多项目索引
│
├─ 项目 1
│  ├─ Python 文件 → CodeChunker → 标注 + LLM 摘要
│  ├─ C++ 文件 → CodeChunker → 脱敏 + LLM 摘要 → 敏感信息检测
│  ├─ Verilog 文件 → VerilogChunker → 标注 + LLM 摘要
│  ├─ 文档 → DocChunker → 标注
│  ├─ 构建知识图谱 → Entity/Relation 提取 + 跨文件解析
│  └─ 生成项目摘要 → PROJECT_INDEX（两阶段检索入口）
│
├─ 项目 2
│  └─ （同上）
│
└─ 查询
   ├─ Embedding 查询（本地 Embedding API）
   ├─ PROJECT_INDEX 检索（Top-k 项目）
   ├─ CODE_INDEX + DOC_INDEX 检索（按 project_id 过滤）
   ├─ 知识图谱扩展（可选：向量检索 + 图 BFS 扩展）
   ├─ 组装块 + 元数据
   └─ 发送给本地 LLM API（Qwen3.5-8B 或兼容模型）
```

### 索引结构

| 索引 | 内容 | 用途 |
|------|------|------|
| `code_index` | 函数/类/模块块，含项目/语言元数据 | 代码检索 |
| `doc_index` | Markdown/Word 文档块 | 文档检索 |
| `project_index` | 项目级摘要 | 第一阶段检索入口 |
| `knowledge_graph.json` | NetworkX 节点链接格式 | 图谱查询与可视化 |
| `file_hashes.json` | 文件路径 → SHA-256 | 增量更新追踪 |

### 知识图谱模块结构

```
src/sec_rag/graph/
├── __init__.py              # 公开 API
├── backend.py               # GraphBackend 抽象基类
├── models.py                # Entity / EntityId / Relation 数据模型
├── ast_extractor.py         # tree-sitter AST 实体 + 关系提取
│   └── SymbolTable          # 项目级符号表（跨文件解析）
├── llm_extractor.py         # LLM fallback（AST 失败时）
├── networkx_backend.py      # NetworkX 后端实现
├── store.py                 # GraphStore 门面（统一接口）
├── builder.py               # 增量构建 + build_project()
├── file_hash_index.py       # SHA-256 文件哈希追踪
├── query.py                 # 高层查询（callers/callees/hierarchy/impact）
├── exporter.py              # DOT / GEXF / JSON 导出 + 命名空间聚类
└── html_exporter.py         # 交互式 D3.js HTML 导出
```

---

## 知识图谱设计

### 实体类型（Entity Types）

| 类型 | 说明 |
|------|------|
| `file` | 源代码文件 |
| `function` | 函数/方法 |
| `class` | 类定义 |
| `struct` | 结构体 |
| `namespace` | 命名空间 |
| `template` | 模板定义 |
| `type_alias` | 类型别名（typedef / using） |
| `concept` | C++20 concept |

### 关系类型（Relation Types）

| 关系 | 方向 | 说明 |
|------|------|------|
| `CONTAINS` | file → entity | 文件包含实体 |
| `INCLUDES` | file → file | 文件包含头文件 |
| `CALLS` | function → function | 函数调用函数 |
| `REFERENCES` | entity → type | 实体引用某类型 |
| `INHERITS_FROM` | class → class | 类继承自某类 |

---

## 隐私模式设计

### 数据模型

Chunk 元数据扩展：
| 字段 | 类型 | 说明 |
|------|------|------|
| `is_private` | `bool` | 是否为高密级文件 |
| `privacy_summary` | `str` | LLM 生成的隐私安全摘要 |

### 索引流程

1. **配置加载**：自动检测 `.privacy_config.json` 或环境变量 `PRIVACY_CONFIG_PATH`
2. **文件检测**：对每个文件调用 `PrivacyConfig.is_private()`
3. **摘要生成**：对隐私文件调用 `generate_privacy_summary()`（LLM 生成）
4. **元数据标记**：设置 `is_private=True`，存入 `privacy_summary`
5. **原文保留**：原始内容仍存入向量库用于检索，但不会被发送给 LLM

### 查询流程

1. **检索阶段**：正常检索（隐私文件也参与向量相似度计算）
2. **检测日志**：如果检索结果包含隐私文件，打印 `[PRIVACY ALERT]` 警告
3. **Prompt 构建**：自动将隐私文件的 `text` 替换为 `privacy_summary`
4. **LLM 调用**：Prompt 中绝不包含隐私文件原文

---

## 测试

```bash
# 运行所有测试
pytest -q

# 或
uv run pytest -q

# 运行特定测试文件
pytest tests/test_graph.py -v
pytest tests/test_graph_phase2.py -v
pytest tests/test_graph_phase3.py -v
pytest tests/test_graph_phase4.py -v

# 运行特定测试类
pytest tests/test_code_chunker.py::TestPythonFunctionExtraction
```

### 测试覆盖

| 测试文件 | 说明 | 用例数 |
|----------|------|--------|
| `tests/test_code_chunker.py` | Python/C 代码分块 | 18 |
| `tests/test_graph.py` | 知识图谱 Phase 1（MVP） | 19 |
| `tests/test_graph_phase2.py` | 深层关系 + 混合检索 | 10 |
| `tests/test_graph_phase3.py` | 增量更新 + 抽象层 | 10 |
| `tests/test_graph_phase4.py` | 哈希追踪 + Watch + HTML | 9 |
| `tests/test_privacy.py` | 隐私模式配置 + 摘要 + Prompt 替换 | 15 |
| **总计** | | **81** |

---

## 性能说明

- **首次运行**：模型下载 + ChromaDB 初始化（约 5-10 分钟，取决于模型大小）
- **索引速度**：约 100-200 文件/分钟（取决于文件大小、LLM 摘要延迟）
- **查询延迟**：约 1-2 秒（本地 Embedding + ChromaDB 检索）+ LLM 生成时间
- **Token 节省**：相比整文件检索节省 60-80%（脱敏摘要 100-150 tokens vs 完整函数 2000+ tokens）
- **图谱规模**：测试项目（23 个 C++ 文件）→ 260 节点 / 485 边

---

## 故障排除

**错误：`ModuleNotFoundError: No module named 'chromadb'`**
```bash
pip install -e ".[dev]"
```

**错误：本地 LLM 初始化失败**
- 确保本地模型网关正在运行
- 检查 `--api-base-url` 指向正确的端点
- 验证端点兼容 OpenAI 或 Anthropic API

**错误：索引时抛出 `SensitiveInfoError`**
- 查看被标记的文档片段
- 如为误报，请提交 issue
- 敏感模式检测为硬失败策略（设计如此）

**错误：tree-sitter 解析失败**
- 确保已安装语言库：`pip install tree-sitter-python tree-sitter-c tree-sitter-cpp`
- C++ 文件可能包含无法解析的宏，系统会自动回退到 LLM 提取

**隐私文件被检索但摘要未显示**
- 确认 `.privacy_config.json` 已正确放置并在索引时加载（查看 `[PRIVACY]` 日志）
- 隐私摘要只在发送到 LLM 的 Prompt 中替换，数据库中仍存储原文用于检索
- 使用 `--verbose` 查看详细的隐私检测日志

**隐私摘要生成失败**
- 确保 LLM 服务在索引时可用（摘要是在索引阶段生成的）
- 检查 LLM 超时设置：`--llm-timeout 300`
- 如 LLM 失败，系统会回退到默认模板摘要

---

## 许可证

MIT
