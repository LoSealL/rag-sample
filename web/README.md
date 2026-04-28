# sec-rag Web UI

基于 Vue 3 + TypeScript + FastAPI 的 sec-rag 主观测试平台。

## 功能特性

- 📤 **代码上传**：支持文件/文件夹拖拽、压缩包自动解压
- 🚀 **建库进度**：实时 SSE 进度条，建库完成后自动跳转查询页
- 💬 **RAG 查询**：输入即搜索（防抖 500ms），展示匹配代码块和 LLM 回答
- ⚙️ **配置面板**：匹配方式（向量/图谱/混合）、Top-K、分块参数、模型设置
- 📊 **测评可视化**：上传 reference_labels.json，以 Diff 形式展示匹配/不匹配结果

## 技术栈

### 后端
- FastAPI + Uvicorn
- SSE (Server-Sent Events) 实时进度推送
- 复用 sec_rag 核心逻辑（索引、查询、分块）

### 前端
- Vue 3 (Composition API)
- TypeScript
- Vite
- Pinia (状态管理)
- Axios (HTTP 客户端)

## 目录结构

```
web/
├── backend/              # FastAPI 后端
│   ├── main.py           # 应用入口
│   ├── api/
│   │   ├── upload.py     # 文件上传/解压
│   │   ├── index.py      # 建库 (SSE 进度)
│   │   ├── query.py      # RAG 查询
│   │   ├── config.py     # 配置管理
│   │   └── eval.py       # 测评 Diff
│   └── requirements.txt
├── frontend/             # Vue 3 前端
│   ├── src/
│   │   ├── main.ts       # 入口
│   │   ├── App.vue       # 主布局
│   │   ├── api/
│   │   │   └── client.ts # API 客户端
│   │   ├── stores/
│   │   │   └── app.ts    # Pinia store
│   │   └── views/
│   │       ├── UploadView.vue   # 上传/建库
│   │       ├── QueryView.vue    # RAG 查询
│   │       ├── ConfigView.vue   # 配置面板
│   │       └── EvalView.vue     # 测评可视化
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── start_backend.bat     # Windows 后端启动脚本
└── start_frontend.bat    # Windows 前端启动脚本
```

## 快速开始

### 1. 启动后端

```bash
# Windows
.\start_backend.bat

# 或手动
cd web/backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

后端服务运行在 http://localhost:8001

### 2. 启动前端

```bash
# Windows
.\start_frontend.bat

# 或手动
cd web/frontend
npm install
npm run dev
```

前端服务运行在 http://localhost:5173

### 3. 使用

1. 打开浏览器访问 http://localhost:5173
2. 在"上传代码"页拖拽文件/文件夹
3. 点击"开始建库"，等待进度完成
4. 自动跳转到"RAG 查询"页，输入问题即可实时查询

## API 文档

启动后端后访问 http://localhost:8001/docs 查看自动生成的 Swagger UI。

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload/files` | 上传文件/压缩包 |
| GET | `/api/index/build-sse` | 建库（SSE 进度流） |
| POST | `/api/query/rag` | RAG 查询 |
| GET/POST | `/api/config/` | 获取/更新配置 |
| POST | `/api/eval/upload` | 上传测评 JSON |

## 注意事项

1. **后端依赖**：需要安装 sec_rag 的 Python 依赖（`pip install -e .`）
2. **模型服务**：确保本地 LLM 和 Embedding 服务已启动（默认 http://localhost:8000）
3. **隐私模式**：Web UI 目前未集成隐私配置，如需使用请先通过 CLI 建库
4. **测评 JSON**：支持 `reference_labels.json` 格式，需包含 `file`, `name`, `type`, `line_start`, `line_end` 字段

## 开发计划

- [ ] 集成隐私模式到 Web UI
- [ ] 支持图谱查询可视化
- [ ] 支持多项目切换
- [ ] 查询历史记录
- [ ] 导出测评报告
