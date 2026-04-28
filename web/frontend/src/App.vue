<template>
  <div class="app">
    <header class="header">
      <h1>🧠 sec-rag Web UI</h1>
      <p class="subtitle">代码知识图谱与 RAG 主观测试平台</p>
    </header>

    <nav class="nav">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="['nav-btn', { active: store.activeTab === tab.id }]"
        @click="store.setActiveTab(tab.id)"
        :disabled="tab.disabled && !store.canQuery"
      >
        {{ tab.label }}
        <span v-if="tab.id === 'query' && store.isIndexed" class="badge">✓</span>
      </button>
    </nav>

    <main class="main">
      <UploadView v-if="store.activeTab === 'upload'" />
      <QueryView v-else-if="store.activeTab === 'query'" />
      <ConfigView v-else-if="store.activeTab === 'config'" />
      <EvalView v-else-if="store.activeTab === 'eval'" />
    </main>

    <footer class="footer">
      <span v-if="store.currentProjectId">
        项目: {{ store.currentProjectName }} ({{ store.currentProjectId }})
        <span v-if="store.isIndexed" class="status-ok">[已建库]</span>
        <span v-else-if="store.isIndexing" class="status-progress">[建库中...]</span>
        <span v-else class="status-pending">[待建库]</span>
      </span>
      <span v-else>未选择项目</span>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from '@/stores/app'
import UploadView from '@/views/UploadView.vue'
import QueryView from '@/views/QueryView.vue'
import ConfigView from '@/views/ConfigView.vue'
import EvalView from '@/views/EvalView.vue'

const store = useAppStore()

const tabs = [
  { id: 'upload', label: '📤 上传代码', disabled: false },
  { id: 'query', label: '💬 RAG 查询', disabled: true },
  { id: 'config', label: '⚙️ 配置', disabled: false },
  { id: 'eval', label: '📊 测评', disabled: false },
]
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 24px 32px;
  text-align: center;
}

.header h1 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 8px;
}

.subtitle {
  font-size: 14px;
  opacity: 0.9;
}

.nav {
  display: flex;
  background: white;
  border-bottom: 1px solid #e2e8f0;
  padding: 0 32px;
  gap: 4px;
}

.nav-btn {
  padding: 14px 24px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: #64748b;
  border-bottom: 3px solid transparent;
  transition: all 0.2s;
  position: relative;
}

.nav-btn:hover:not(:disabled) {
  color: #4f46e5;
  background: #f8fafc;
}

.nav-btn.active {
  color: #4f46e5;
  border-bottom-color: #4f46e5;
}

.nav-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.badge {
  margin-left: 6px;
  color: #22c55e;
}

.main {
  flex: 1;
  padding: 24px 32px;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
}

.footer {
  background: white;
  border-top: 1px solid #e2e8f0;
  padding: 12px 32px;
  font-size: 13px;
  color: #64748b;
  display: flex;
  justify-content: space-between;
}

.status-ok { color: #22c55e; font-weight: 600; }
.status-progress { color: #f59e0b; font-weight: 600; }
.status-pending { color: #ef4444; font-weight: 600; }
</style>
