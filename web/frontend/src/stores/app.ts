import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAppStore = defineStore('app', () => {
  // State
  const currentProjectId = ref('')
  const currentProjectName = ref('')
  const isIndexed = ref(false)
  const isIndexing = ref(false)
  const indexProgress = ref({ current: 0, total: 0, message: '' })
  const config = ref({
    match_mode: 'vector',
    top_k: 5,
    project_top_k: 3,
    chunk_backend: 'ast',
    chunk_naive_lines: 60,
    chunk_overlap_ratio: 0.2,
    chunk_max_tokens: 600,
    embedding_model: 'Qwen3-VL-Embedding-8B',
    llm_model: 'Qwen3.5-8B',
    llm_provider: 'auto',
    api_base_url: 'http://localhost:8000',
    enable_rerank: false,
    rerank_model: 'BAAI/bge-reranker-base',
  })
  const activeTab = ref('upload')

  // Getters
  const canIndex = computed(() => !!currentProjectId.value && !isIndexed.value && !isIndexing.value)
  const canQuery = computed(() => isIndexed.value)

  // Actions
  function setProject(id: string, name: string) {
    currentProjectId.value = id
    currentProjectName.value = name
    isIndexed.value = false
  }

  function setIndexed(value: boolean) {
    isIndexed.value = value
  }

  function setIndexing(value: boolean) {
    isIndexing.value = value
  }

  function setProgress(current: number, total: number, message: string) {
    indexProgress.value = { current, total, message }
  }

  function updateConfig(partial: Partial<typeof config.value>) {
    config.value = { ...config.value, ...partial }
  }

  function setActiveTab(tab: string) {
    activeTab.value = tab
  }

  return {
    currentProjectId,
    currentProjectName,
    isIndexed,
    isIndexing,
    indexProgress,
    config,
    activeTab,
    canIndex,
    canQuery,
    setProject,
    setIndexed,
    setIndexing,
    setProgress,
    updateConfig,
    setActiveTab,
  }
})
