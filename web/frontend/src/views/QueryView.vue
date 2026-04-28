<template>
  <div class="query-view">
    <section class="query-section">
      <h2>💬 RAG 查询</h2>      
      <div class="query-input-wrapper">
        <input
          v-model="queryText"
          class="query-input"
          placeholder="输入你的问题，例如：DMA 控制器是如何初始化的？"
          :disabled="!store.canQuery"
          @input="handleInput"
        />
        <button
          class="btn-send"
          :disabled="!store.canQuery || !queryText"
          @click="executeQuery"
        >
          🔍 查询
        </button>
      </div>

      <div v-if="isLoading" class="loading">
        <span class="spinner">⏳</span> 正在检索...
      </div>
    </section>

    <section v-if="chunks.length > 0" class="results-section">
      <h3>📄 匹配到的代码块 (Top-{{ chunks.length }})</h3>
      <div class="chunks-list">
        <div
          v-for="(chunk, idx) in chunks"
          :key="chunk.chunk_id"
          class="chunk-card"
          :class="{ private: chunk.metadata?.is_private }"
        >
          <div class="chunk-header">
            <span class="chunk-index">#{{ idx + 1 }}</span>
            <span class="chunk-score">Score: {{ chunk.score.toFixed(4) }}</span>
            <span v-if="chunk.metadata?.is_private" class="privacy-badge">🔒 隐私</span>
          </div>
          <div class="chunk-meta">
            {{ chunk.metadata?.file_path }} | Lines {{ chunk.metadata?.line_start }}-{{ chunk.metadata?.line_end }}
          </div>
          <pre class="chunk-code"><code>{{ chunk.text }}</code></pre>
        </div>
      </div>
    </section>

    <section v-if="answer" class="answer-section">
      <h3>🤖 LLM 回答</h3>
      <div class="answer-box">
        <pre>{{ answer }}</pre>
      </div>
      <div class="token-info">
        Prompt Tokens: {{ totalPromptTokens }} | Chunk Tokens: {{ totalChunkTokens }}
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useAppStore } from '@/stores/app'
import { queryRag } from '@/api/client'
import type { ChunkResult } from '@/api/client'

const store = useAppStore()
const queryText = ref('')
const isLoading = ref(false)
const chunks = ref<ChunkResult[]>([])
const answer = ref('')
const totalPromptTokens = ref(0)
const totalChunkTokens = ref(0)

let debounceTimer: ReturnType<typeof setTimeout> | null = null

function handleInput() {
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(() => {
    if (queryText.value.length > 3) {
      executeQuery()
    }
  }, 500)
}

async function executeQuery() {
  if (!store.canQuery || !queryText.value) return

  isLoading.value = true
  try {
    const result = await queryRag({
      project_id: store.currentProjectId,
      query: queryText.value,
      top_k: store.config.top_k,
      filter_language: '',
      filter_chunk_type: '',
      project_top_k: store.config.project_top_k,
    })

    chunks.value = result.chunks
    answer.value = result.answer
    totalPromptTokens.value = result.total_prompt_tokens
    totalChunkTokens.value = result.total_chunk_tokens
  } catch (err) {
    alert('查询失败: ' + (err as Error).message)
  } finally {
    isLoading.value = false
  }
}

// Auto query when tab becomes active and query exists
watch(() => store.activeTab, (tab) => {
  if (tab === 'query' && queryText.value.length > 3 && store.canQuery) {
    executeQuery()
  }
})
</script>

<style scoped>
.query-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.query-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.query-section h2 {
  margin-bottom: 16px;
  font-size: 18px;
  color: #1e293b;
}

.query-input-wrapper {
  display: flex;
  gap: 12px;
}

.query-input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.query-input:focus {
  outline: none;
  border-color: #4f46e5;
}

.query-input:disabled {
  background: #f1f5f9;
  cursor: not-allowed;
}

.btn-send {
  padding: 12px 24px;
  background: #4f46e5;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
}

.btn-send:hover:not(:disabled) {
  background: #4338ca;
}

.btn-send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading {
  margin-top: 16px;
  padding: 12px;
  text-align: center;
  color: #64748b;
}

.spinner {
  display: inline-block;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.results-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.results-section h3 {
  margin-bottom: 16px;
  font-size: 16px;
  color: #1e293b;
}

.chunks-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 16px;
  background: #f8fafc;
}

.chunk-card.private {
  border-color: #f59e0b;
  background: #fffbeb;
}

.chunk-header {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 8px;
}

.chunk-index {
  font-weight: 600;
  color: #4f46e5;
}

.chunk-score {
  font-size: 12px;
  color: #64748b;
  background: #e0e7ff;
  padding: 2px 8px;
  border-radius: 4px;
}

.privacy-badge {
  font-size: 12px;
  color: #92400e;
  background: #fef3c7;
  padding: 2px 8px;
  border-radius: 4px;
}

.chunk-meta {
  font-size: 12px;
  color: #94a3b8;
  margin-bottom: 8px;
}

.chunk-code {
  background: #1e293b;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.5;
}

.answer-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.answer-section h3 {
  margin-bottom: 16px;
  font-size: 16px;
  color: #1e293b;
}

.answer-box {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 8px;
  padding: 16px;
}

.answer-box pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 14px;
  line-height: 1.6;
  color: #166534;
}

.token-info {
  margin-top: 12px;
  font-size: 12px;
  color: #94a3b8;
  text-align: right;
}
</style>
