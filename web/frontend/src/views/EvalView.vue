<template>
  <div class="eval-view">
    <section class="upload-section">
      <h2>📊 测评可视化</h2>      
      <div class="upload-area">
        <input
          ref="fileInput"
          type="file"
          accept=".json"
          style="display: none"
          @change="handleFileSelect"
        />
        <button class="btn-primary" @click="fileInput?.click()">
          📤 上传标注 JSON
        </button>
        <p class="hint">支持 reference_labels.json 格式</p>
      </div>
    </section>

    <section v-if="summary" class="summary-section">
      <h3>测评概览</h3>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">{{ summary.total }}</div>
          <div class="stat-label">总词条</div>
        </div>
        <div class="stat-card">
          <div class="stat-value success">{{ summary.matched }}</div>
          <div class="stat-label">匹配</div>
        </div>
        <div class="stat-card">
          <div class="stat-value error">{{ summary.total - summary.matched }}</div>
          <div class="stat-label">不匹配</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" :class="accuracyClass">
            {{ (summary.accuracy * 100).toFixed(1) }}%
          </div>
          <div class="stat-label">准确率</div>
        </div>
      </div>
    </section>

    <section v-if="items.length > 0" class="diff-section">
      <h3>Diff 详情</h3>
      <div class="diff-list">
        <div
          v-for="item in items"
          :key="item.name + item.file"
          class="diff-item"
          :class="{ matched: item.match, mismatched: !item.match }"
        >
          <div class="diff-header">
            <span class="diff-name">{{ item.name }}</span>
            <span :class="['diff-badge', item.match ? 'badge-success' : 'badge-error']">
              {{ item.match ? '✓ 匹配' : '✗ 不匹配' }}
            </span>
          </div>
          <div class="diff-body">
            <div class="diff-side expected">
              <div class="side-label">期望</div>
              <div class="side-content">
                <p><strong>类型:</strong> {{ item.expected_type }}</p>
                <p><strong>行范围:</strong> {{ item.expected_start }} - {{ item.expected_end }}</p>
              </div>
            </div>
            <div class="diff-side predicted">
              <div class="side-label">预测</div>
              <div class="side-content">
                <p>
                  <strong>类型:</strong>
                  <span :class="{ diff: item.expected_type !== item.predicted_type }">
                    {{ item.predicted_type || 'N/A' }}
                  </span>
                </p>
                <p>
                  <strong>行范围:</strong>
                  <span :class="{ diff: item.expected_start !== item.predicted_start || item.expected_end !== item.predicted_end }">
                    {{ item.predicted_start ?? 'N/A' }} - {{ item.predicted_end ?? 'N/A' }}
                  </span>
                </p>
              </div>
            </div>
          </div>
          <div class="diff-footer">
            📁 {{ item.file }}
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { uploadEval } from '@/api/client'
import type { EvalSummary, EvalItem } from '@/api/client'

const fileInput = ref<HTMLInputElement | null>(null)
const summary = ref<EvalSummary | null>(null)
const items = ref<EvalItem[]>([])

const accuracyClass = computed(() => {
  if (!summary.value) return ''
  const acc = summary.value.accuracy
  if (acc >= 0.9) return 'success'
  if (acc >= 0.7) return 'warning'
  return 'error'
})

async function handleFileSelect(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return

  try {
    const result = await uploadEval(file)
    summary.value = result
    items.value = result.items
  } catch (err) {
    alert('上传失败: ' + (err as Error).message)
  }
}
</script>

<style scoped>
.eval-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.upload-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.upload-section h2 {
  margin-bottom: 16px;
  font-size: 18px;
  color: #1e293b;
}

.upload-area {
  text-align: center;
  padding: 32px;
  border: 2px dashed #e2e8f0;
  border-radius: 8px;
}

.hint {
  margin-top: 8px;
  color: #94a3b8;
  font-size: 13px;
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

.summary-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 16px;
}

.stat-card {
  text-align: center;
  padding: 20px;
  background: #f8fafc;
  border-radius: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #1e293b;
}

.stat-value.success { color: #22c55e; }
.stat-value.error { color: #ef4444; }
.stat-value.warning { color: #f59e0b; }

.stat-label {
  margin-top: 4px;
  font-size: 13px;
  color: #64748b;
}

.diff-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.diff-section h3 {
  margin-bottom: 16px;
}

.diff-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.diff-item {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.diff-item.matched {
  border-color: #bbf7d0;
}

.diff-item.mismatched {
  border-color: #fecaca;
}

.diff-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.diff-name {
  font-weight: 600;
  font-size: 14px;
}

.diff-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.badge-success {
  background: #dcfce7;
  color: #166534;
}

.badge-error {
  background: #fee2e2;
  color: #991b1b;
}

.diff-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}

.diff-side {
  padding: 16px;
}

.diff-side.expected {
  background: #f0fdf4;
  border-right: 1px solid #e2e8f0;
}

.diff-side.predicted {
  background: #fef2f2;
}

.side-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 8px;
}

.side-content {
  font-size: 13px;
  line-height: 1.6;
}

.side-content p {
  margin: 4px 0;
}

.diff {
  background: #fee2e2;
  padding: 0 4px;
  border-radius: 2px;
  color: #991b1b;
}

.diff-footer {
  padding: 8px 16px;
  font-size: 12px;
  color: #94a3b8;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
}
</style>
