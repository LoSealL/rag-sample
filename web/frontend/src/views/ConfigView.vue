<template>
  <div class="config-view">
    <section class="config-section">
      <h2>⚙️ 配置面板</h2>      
      <form @submit.prevent="saveConfig">
        <!-- Match Mode -->
        <div class="form-group">
          <label>匹配方式</label>
          <div class="radio-group">
            <label v-for="mode in matchModes" :key="mode.value">
              <input
                type="radio"
                v-model="localConfig.match_mode"
                :value="mode.value"
              />
              {{ mode.label }}
            </label>
          </div>
        </div>

        <!-- Top-K -->
        <div class="form-group">
          <label>Top-K: {{ localConfig.top_k }}</label>
          <input
            type="range"
            v-model.number="localConfig.top_k"
            min="1"
            max="20"
          />
        </div>

        <!-- Chunk Backend -->
        <div class="form-group">
          <label>分块后端</label>
          <select v-model="localConfig.chunk_backend">
            <option value="ast">AST (语法树)</option>
            <option value="llm">LLM (大模型)</option>
            <option value="hybrid">Hybrid (混合)</option>
          </select>
        </div>

        <!-- Chunk Lines -->
        <div class="form-group">
          <label>分块行数: {{ localConfig.chunk_naive_lines }}</label>
          <input
            type="range"
            v-model.number="localConfig.chunk_naive_lines"
            min="10"
            max="200"
            step="10"
          />
        </div>

        <!-- Overlap Ratio -->
        <div class="form-group">
          <label>重叠比例: {{ (localConfig.chunk_overlap_ratio * 100).toFixed(0) }}%</label>
          <input
            type="range"
            v-model.number="localConfig.chunk_overlap_ratio"
            min="0"
            max="0.5"
            step="0.05"
          />
        </div>

        <!-- Max Tokens -->
        <div class="form-group">
          <label>最大 Token: {{ localConfig.chunk_max_tokens }}</label>
          <input
            type="range"
            v-model.number="localConfig.chunk_max_tokens"
            min="200"
            max="2000"
            step="100"
          />
        </div>

        <!-- Model Settings -->
        <div class="form-group">
          <label>Embedding 模型</label>
          <input v-model="localConfig.embedding_model" type="text" />
        </div>

        <div class="form-group">
          <label>LLM 模型</label>
          <input v-model="localConfig.llm_model" type="text" />
        </div>

        <div class="form-group">
          <label>LLM Provider</label>
          <select v-model="localConfig.llm_provider">
            <option value="auto">Auto</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
          </select>
        </div>

        <div class="form-group">
          <label>API Base URL</label>
          <input v-model="localConfig.api_base_url" type="text" />
        </div>

        <!-- Rerank -->
        <div class="form-group">
          <label class="checkbox-label">
            <input type="checkbox" v-model="localConfig.enable_rerank" />
            启用重排序 (Rerank)
          </label>
        </div>

        <div class="actions">
          <button type="submit" class="btn-primary">💾 保存配置</button>
          <button type="button" class="btn-secondary" @click="resetConfig">重置</button>
        </div>
      </form>
    </section>
  </div>
</template>

<script setup lang="ts">
import { reactive, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { getConfig, updateConfig } from '@/api/client'

const store = useAppStore()

const matchModes = [
  { value: 'vector', label: '🔍 向量检索' },
  { value: 'graph', label: '🕸️ 图谱检索' },
  { value: 'hybrid', label: '🔀 混合检索' },
]

const localConfig = reactive({ ...store.config })

onMounted(async () => {
  try {
    const cfg = await getConfig()
    Object.assign(localConfig, cfg)
  } catch (err) {
    console.error('Failed to load config:', err)
  }
})

async function saveConfig() {
  try {
    const updated = await updateConfig({ ...localConfig })
    store.updateConfig(updated)
    alert('配置已保存！')
  } catch (err) {
    alert('保存失败: ' + (err as Error).message)
  }
}

function resetConfig() {
  Object.assign(localConfig, store.config)
}
</script>

<style scoped>
.config-view {
  max-width: 600px;
}

.config-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.config-section h2 {
  margin-bottom: 20px;
  font-size: 18px;
  color: #1e293b;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #374151;
}

.form-group input[type="text"],
.form-group select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
}

.form-group input[type="range"] {
  width: 100%;
  margin-top: 4px;
}

.radio-group {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.radio-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: normal;
  cursor: pointer;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #e5e7eb;
}

.btn-primary {
  padding: 10px 24px;
  background: #4f46e5;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.btn-primary:hover {
  background: #4338ca;
}

.btn-secondary {
  padding: 10px 24px;
  background: white;
  color: #64748b;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
}
</style>
