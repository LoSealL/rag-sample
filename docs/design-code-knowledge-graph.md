# 设计提案：代码知识图谱（Code Knowledge Graph）模块

## 1. 背景与动机

当前 `sec-rag` 的检索粒度是 **chunk 级别**（函数/类/文档段落），通过向量相似度找到最相关的代码块。这种方式在回答"某个函数是做什么的"时效果很好，但在以下场景存在局限：

- **调用链分析**："`authenticate_user` 被哪些函数调用了？"
- **依赖影响评估**："修改 `Document` 结构会影响哪些模块？"
- **架构理解**："从用户请求到数据存储的完整调用链路是什么？"
- **概念关联**："哪些类实现了 `EmbeddingModel` concept？"

**目标**：在现有 chunk 索引之上，构建一个**代码实体关系图谱**，支持图遍历查询，并与向量检索形成互补。

---

## 2. 核心概念

### 2.1 实体（Nodes）

从 chunk 中提取的代码实体：

| 实体类型 | 来源 | 示例 |
|---------|------|------|
| `File` | 文件路径 | `include/rag/core/types.hpp` |
| `Namespace` | namespace 定义 | `rag` |
| `Class` / `Struct` | class/struct 定义 | `Document`, `Chunk` |
| `Function` | function 定义 | `mean_vector`, `l2_norm` |
| `TypeAlias` | using 定义 | `scalar_t`, `dense_vector_t` |
| `Concept` | concept 定义 | `EmbeddingModel` |
| `Variable` | 全局/静态变量 | `g_instance` |
| `Macro` | #define | `ASSERT` |
| `Project` | 项目根目录 | `rag-test` |

每个实体包含：
- `id`: 全局唯一标识（基于 project + file + name + type 的 hash）
- `name`: 符号名
- `type`: 实体类型
- `file_path`: 所在文件
- `line_start`, `line_end`: 代码位置
- `chunk_id`: 关联的 chunk（用于回查原文）
- `declaration`: 声明签名（用于快速预览）
- `project_id`: 所属项目

### 2.2 关系（Edges）

#### 文件层（编译单元级）

| 关系类型 | 方向 | 说明 |
|---------|------|------|
| `INCLUDES` | File → File | `#include "xxx.hpp"` |
| `INCLUDED_BY` | File → File | 被包含（反向） |

#### 符号层（语义级）

| 关系类型 | 方向 | 说明 |
|---------|------|------|
| `CONTAINS` | Namespace/Class → Class/Function/TypeAlias | 包含关系 |
| `INHERITS_FROM` | Class → Class | `class A : public B` |
| `IMPLEMENTS` | Class → Concept | `class A : EmbeddingModel` |
| `CALLS` | Function → Function | 函数调用 |
| `REFERENCES` | Function/Class → TypeAlias/Class | 类型引用（参数、返回值、成员变量） |
| `OVERRIDES` | Function → Function | 虚函数重写 |
| `INSTANTIATES` | Function/File → Class | 模板实例化 |
| `USES_CONCEPT` | Function/Class → Concept | requires 约束 |
| `DEFINED_IN` | 任意符号 → File | 定义所在文件 |
| `BELONGS_TO` | 任意符号 → Project | 所属项目 |

---

## 3. 系统架构

```
                    ┌─────────────────────────────────────┐
                    │         现有 RAG 流程                │
                    │  Index → Chunk → Vector Search     │
                    └──────────────┬──────────────────────┘
                                   │ chunks
                                   ▼
                    ┌─────────────────────────────────────┐
                    │     Code Knowledge Graph Builder     │
                    │  (新增模块: src/sec_rag/graph/)      │
                    └──────────────┬──────────────────────┘
                                   │ entities + relations
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         Graph Store                  │
                    │  NetworkX (内存) + JSON (持久化)      │
                    │  未来可替换为 Neo4j / Kùzu            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
            ┌──────────────┐            ┌─────────────────┐
            │ Graph Query  │            │ Hybrid Query    │
            │  API         │            │  (Vector+Graph) │
            │              │            │                 │
            │ - neighbors  │            │ - 先用向量检索   │
            │ - path       │            │   候选 chunks   │
            │ - impact     │            │ - 再沿图谱扩展   │
            └──────────────┘            │   相关实体      │
                                        └─────────────────┘
```

---

## 4. 模块设计

### 4.1 模块结构

```
src/sec_rag/
├── graph/
│   ├── __init__.py
│   ├── models.py              # Entity, Relation, GraphNode, GraphEdge 数据类
│   ├── extractor.py           # 从 chunk/source 中提取实体和关系
│   ├── ast_extractor.py       # 基于 tree-sitter AST 的精确提取
│   ├── llm_extractor.py       # 基于 LLM 的模糊提取（fallback）
│   ├── builder.py             # 构建完整图谱
│   ├── store.py               # 图存储接口
│   ├── query.py               # 图查询接口
│   └── exporter.py            # 导出为 DOT / GEXF / Cypher
```

### 4.2 关键类设计

#### `models.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EntityId:
    """全局唯一实体标识。"""
    project_id: str
    file_path: str
    name: str
    entity_type: str  # "class", "function", "type_alias", ...
    
    def __str__(self) -> str:
        return f"{self.project_id}:{self.file_path}:{self.entity_type}:{self.name}"


@dataclass
class Entity:
    """代码实体节点。"""
    id: EntityId
    name: str
    entity_type: str
    file_path: str
    line_start: int
    line_end: int
    chunk_id: str = ""
    declaration: str = ""
    project_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Relation:
    """实体之间的关系边。"""
    source: EntityId
    target: EntityId
    relation_type: str  # "CALLS", "INHERITS_FROM", "INCLUDES", ...
    file_path: str = ""  # 关系发生的文件
    line: int = 0        # 关系发生的行号
    metadata: dict = field(default_factory=dict)
```

#### `extractor.py`

```python
from abc import ABC, abstractmethod
from typing import Iterator

from .models import Entity, Relation


class GraphExtractor(ABC):
    """从源代码中提取实体和关系的抽象接口。"""
    
    @abstractmethod
    def extract_entities(self, source: str, file_path: str, project_id: str) -> Iterator[Entity]:
        """提取所有实体。"""
        
    @abstractmethod
    def extract_relations(self, source: str, file_path: str, project_id: str) -> Iterator[Relation]:
        """提取所有关系。"""
```

#### `ast_extractor.py`

使用 tree-sitter AST 实现精确提取：

- **实体提取**：遍历 AST，识别 `class_specifier`、`struct_specifier`、`function_definition`、`alias_declaration`、`namespace_definition`、`concept_definition`
- **关系提取**：
  - `INCLUDES`：扫描 `#include` 预处理指令
  - `INHERITS_FROM`：从 `class_specifier` 的 `base_class_clause` 提取
  - `CONTAINS`：从 `namespace_definition` / `class_specifier` 的 `declaration_list` 提取
  - `CALLS`：从 `function_definition` 体内扫描 `call_expression`
  - `REFERENCES`：从函数参数、返回值、成员变量声明中提取类型引用

#### `llm_extractor.py`

对于 AST 难以解析的复杂场景（宏、模板元编程、跨文件间接引用），使用 LLM 辅助提取：

```python
class LLMGraphExtractor(GraphExtractor):
    def __init__(self, llm_client):
        self._llm = llm_client
    
    def extract_entities(self, source, file_path, project_id):
        prompt = build_entity_extraction_prompt(source, file_path)
        raw = self._llm.complete(prompt)
        yield from parse_entities(raw, file_path, project_id)
    
    def extract_relations(self, source, file_path, project_id):
        prompt = build_relation_extraction_prompt(source, file_path)
        raw = self._llm.complete(prompt)
        yield from parse_relations(raw, file_path, project_id)
```

#### `builder.py`

```python
class GraphBuilder:
    """增量式图谱构建器。"""
    
    def __init__(self, extractor: GraphExtractor, store: GraphStore):
        self._extractor = extractor
        self._store = store
    
    def add_file(self, source: str, file_path: str, project_id: str):
        """向图谱中添加一个文件。"""
        for entity in self._extractor.extract_entities(source, file_path, project_id):
            self._store.add_entity(entity)
        for relation in self._extractor.extract_relations(source, file_path, project_id):
            self._store.add_relation(relation)
    
    def resolve_cross_file_relations(self):
        """二阶段解析：将文件内局部符号解析为全局 EntityId。"""
        # 例如：将 "CALLS print_index_summary" 解析为具体重载版本
```

#### `store.py`

使用 **NetworkX**（纯 Python，无额外服务依赖）作为默认实现：

```python
import json
import networkx as nx
from pathlib import Path

from .models import Entity, Relation, EntityId


class GraphStore:
    """NetworkX-based graph store with JSON persistence."""
    
    def __init__(self, persist_path: str = ".rag_index/knowledge_graph.json"):
        self._persist_path = Path(persist_path)
        self._g = nx.DiGraph()
        self._load()
    
    def add_entity(self, entity: Entity) -> None:
        self._g.add_node(
            str(entity.id),
            **entity.__dict__,
        )
    
    def add_relation(self, relation: Relation) -> None:
        self._g.add_edge(
            str(relation.source),
            str(relation.target),
            relation_type=relation.relation_type,
            file_path=relation.file_path,
            line=relation.line,
            **relation.metadata,
        )
    
    def get_neighbors(
        self, entity_id: EntityId,
        relation_types: list[str] | None = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> list[tuple[Entity, Relation]]:
        """获取邻居实体。"""
        
    def get_paths(
        self, source: EntityId, target: EntityId,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
    ) -> list[list[Entity]]:
        """查找两个实体间的路径。"""
        
    def get_impact_scope(
        self, entity_id: EntityId,
        relation_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        """获取修改某个实体的影响范围。"""
        
    def find_callees(self, entity_id: EntityId, max_depth: int = 3) -> list[Entity]:
        """获取函数/方法的下游调用链。"""
        
    def find_callers(self, entity_id: EntityId, max_depth: int = 3) -> list[Entity]:
        """获取函数/方法的上游调用链。"""
        
    def save(self) -> None:
        """持久化为 JSON。"""
        data = nx.node_link_data(self._g, edges="edges")
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _load(self) -> None:
        if self._persist_path.exists():
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._g = nx.node_link_graph(data, edges="edges")
```

#### `query.py`

```python
class GraphQuery:
    """高层图查询接口。"""
    
    def __init__(self, store: GraphStore):
        self._store = store
    
    def who_calls(self, function_name: str, project_id: str | None = None) -> list[Entity]:
        """哪些函数调用了指定函数？"""
        
    def who_is_called_by(self, function_name: str, project_id: str | None = None) -> list[Entity]:
        """指定函数调用了哪些函数？"""
        
    def class_hierarchy(self, class_name: str, project_id: str | None = None) -> dict:
        """获取类的继承层次。"""
        
    def file_dependencies(self, file_path: str, project_id: str | None = None) -> list[Entity]:
        """获取文件的所有依赖（直接+传递）。"""
        
    def concept_implementations(self, concept_name: str, project_id: str | None = None) -> list[Entity]:
        """哪些类实现了指定 concept？"""
        
    def impact_analysis(self, entity_name: str, project_id: str | None = None) -> dict:
        """修改某实体的影响范围分析。"""
```

---

## 5. 与现有系统的集成点

### 5.1 索引阶段集成

在 `sec-rag index` 流程中，chunking 之后、向量化之前，插入图谱构建步骤：

```python
# cli.py 中的 index 命令
for file_path in files:
    chunks = chunker.chunk_file(file_path)
    
    # 新增：构建知识图谱
    source = Path(file_path).read_text()
    graph_builder.add_file(source, file_path, project.project_id)
    
    # 原有：索引 chunks
    index_store.add_chunks("code_index", chunks, embed_fn=...)

# 所有文件处理完后
graph_builder.resolve_cross_file_relations()
graph_builder.save()
```

### 5.2 查询阶段集成

扩展 `QueryPipeline`，支持混合检索（Hybrid Retrieval）：

```python
class HybridQueryPipeline(QueryPipeline):
    def __init__(self, index_store, graph_store, embedding_model, llm):
        super().__init__(index_store, embedding_model, llm)
        self._graph = graph_store
    
    def run(self, query: str, top_k: int = 5, graph_depth: int = 2):
        # Phase 1: Vector retrieval
        vector_results = super().retrieve(query, top_k=top_k)
        
        # Phase 2: Graph expansion
        graph_results = []
        for chunk in vector_results:
            entity_id = EntityId.from_chunk(chunk)
            neighbors = self._graph.get_neighbors(
                entity_id,
                relation_types=["CALLS", "REFERENCES", "INHERITS_FROM"],
                direction="both",
            )
            for neighbor_entity, relation in neighbors:
                # 获取 neighbor 对应的 chunk
                neighbor_chunk = self._index_store.get_chunk(neighbor_entity.chunk_id)
                if neighbor_chunk:
                    graph_results.append(neighbor_chunk)
        
        # Phase 3: Deduplicate and rank
        all_chunks = self._deduplicate_and_rank(vector_results, graph_results)
        
        # Phase 4: LLM generation
        return self._generate(query, all_chunks)
```

### 5.3 CLI 扩展

新增子命令 `sec-rag graph`：

```bash
# 查看某个函数的调用者
sec-rag graph callers --name l2_norm --project rag-test

# 查看某个类的继承链
sec-rag graph hierarchy --name FlatIndex --project rag-test

# 查看文件依赖图
sec-rag graph deps --file src/store.cpp --project rag-test

# 查看修改影响范围
sec-rag graph impact --name Document --project rag-test

# 导出为 Graphviz DOT
sec-rag graph export --format dot --output deps.dot

# 导出为 GEXF（Gephi 可读）
sec-rag graph export --format gexf --output deps.gexf
```

---

## 6. 实现路线图

### Phase 1: MVP（最小可用）
- [ ] 实现 `models.py`（Entity, Relation 数据类）
- [ ] 实现 `ast_extractor.py`（基于 tree-sitter）
  - 提取实体：class, struct, function, type_alias, concept, namespace
  - 提取关系：INCLUDES, CONTAINS, INHERITS_FROM
- [ ] 实现 `store.py`（NetworkX + JSON）
- [ ] 实现 `query.py`（基础查询：neighbors, paths）
- [ ] CLI 集成：`sec-rag graph` 子命令

### Phase 2: 增强
- [ ] `ast_extractor.py` 增加 CALLS, REFERENCES 提取
- [ ] `llm_extractor.py`（LLM 辅助提取复杂关系）
- [ ] 跨文件关系解析（`resolve_cross_file_relations`）
- [ ] `HybridQueryPipeline`（向量+图混合检索）

### Phase 3: 高级功能
- [ ] 影响范围分析（impact analysis）
- [ ] 可视化导出（DOT, GEXF, Cypher）
- [ ] 增量更新（文件修改时局部更新图谱）
- [ ] 可选后端：Neo4j / Kùzu（用于超大型代码库）

---

## 7. 技术选型理由

| 组件 | 选型 | 理由 |
|------|------|------|
| 图存储 | **NetworkX** (默认) | 纯 Python，零依赖，与现有 ChromaDB 共存；适合中小规模（<10万节点） |
| 图存储 (未来) | **Kùzu** 或 **Neo4j** | 当节点数 >10万 或需要复杂图算法时替换 |
| 关系提取 | **tree-sitter AST** | 已有 CppChunker 使用 tree-sitter，可复用 parser；精度高、速度快 |
| 关系提取 (fallback) | **LLM** | 处理宏、模板、间接引用等 AST 难以分析的场景 |
| 序列化 | **JSON** (NetworkX node_link_data) | 与现有 ChromaDB 同目录存储，便于备份和版本控制 |
| 可视化 | **Graphviz DOT** / **GEXF** | DOT 用于命令行快速查看；GEXF 用于 Gephi 交互式分析 |

---

## 8. 预期收益

1. **检索增强**：向量检索找"语义相似"，图检索找"结构相关"，两者互补
2. **架构理解**：一键生成模块依赖图、调用链路图
3. **安全审计**：快速定位敏感函数（如 `authenticate_user`）的完整调用链
4. **重构辅助**：修改某实体前，一键查看所有影响点
5. **概念对齐**：通过 concept → implementation 关系，理解抽象设计

---

## 9. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| AST 解析失败（语法错误/宏展开） | 关系缺失 | LLM fallback + 优雅降级 |
| 跨文件关系解析不准确 | 错误关联 | 使用 qualified name + 项目级符号表 |
| 大型代码库图谱过大 | 内存/性能 | 支持 Kùzu/Neo4j 后端；按项目隔离 |
| 与现有索引不一致 | 查询结果矛盾 | 共享 chunk_id 作为 bridge；原子化更新 |

---

## 10. 总结

本提案建议在 `sec-rag` 中新增 **Code Knowledge Graph** 模块，利用现有 tree-sitter AST 基础设施提取代码实体和关系，使用 NetworkX 构建轻量级图存储，并通过混合检索增强现有 RAG 能力。该方案与现有架构高度兼容，可作为可选模块逐步集成，不破坏现有功能。

**下一步**：如获批准，将按 Phase 1 MVP 开始实现。
