import axios from 'axios'

const API_BASE = import.meta.env.DEV ? '' : 'http://localhost:8001'

const client = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 300000, // 5 minutes for indexing
  headers: {
    'Content-Type': 'application/json',
  },
})

export default client

export interface UploadResponse {
  project_id: string
  file_count: number
  files: string[]
}

export interface QueryRequest {
  project_id: string
  query: string
  top_k?: number
  filter_language?: string
  filter_chunk_type?: string
  project_top_k?: number
  dry_run?: boolean
}

export interface ChunkResult {
  chunk_id: string
  text: string
  metadata: Record<string, any>
  score: number
}

export interface QueryResponse {
  answer: string
  chunks: ChunkResult[]
  total_prompt_tokens: number
  total_chunk_tokens: number
  candidate_project_ids: string[]
}

export interface Config {
  match_mode: string
  top_k: number
  project_top_k: number
  chunk_backend: string
  chunk_naive_lines: number
  chunk_overlap_ratio: number
  chunk_max_tokens: number
  embedding_model: string
  llm_model: string
  llm_provider: string
  api_base_url: string
  enable_rerank: boolean
  rerank_model: string
}

export interface EvalItem {
  file: string
  name: string
  expected_type: string
  expected_start: number
  expected_end: number
  predicted_type?: string
  predicted_start?: number
  predicted_end?: number
  match: boolean
}

export interface EvalSummary {
  eval_id: string
  total: number
  matched: number
  accuracy: number
  items: EvalItem[]
}

// Upload files
export async function uploadFiles(files: FileList, projectName?: string): Promise<UploadResponse> {
  const formData = new FormData()
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i])
  }
  if (projectName) {
    formData.append('project_name', projectName)
  }
  const { data } = await client.post('/upload/files', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// Build index with SSE (GET endpoint for EventSource compatibility)
export function buildIndex(req: {
  project_id: string
  languages?: string
  chunk_backend?: string
  [key: string]: any
}): EventSource {
  const params = new URLSearchParams()
  Object.entries(req).forEach(([k, v]) => {
    if (v !== undefined) params.append(k, String(v))
  })
  return new EventSource(`${API_BASE}/api/index/build-sse?${params.toString()}`)
}

// Query RAG
export async function queryRag(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await client.post('/query/rag', req)
  return data
}

// Get config
export async function getConfig(): Promise<Config> {
  const { data } = await client.get('/config/')
  return data
}

// Update config
export async function updateConfig(cfg: Partial<Config>): Promise<Config> {
  const { data } = await client.post('/config/', cfg)
  return data
}

// Upload eval JSON
export async function uploadEval(file: File): Promise<EvalSummary> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await client.post('/eval/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
