<template>
  <div class="upload-view">
    <section class="upload-section">
      <h2>📤 上传代码</h2>      
      <div
        class="drop-zone"
        :class="{ dragging: isDragging }"
        @dragover.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="handleDrop"
        @click="fileInput?.click()"
      >
        <input
          ref="fileInput"
          type="file"
          multiple
          webkitdirectory
          directory
          style="display: none"
          @change="handleFileSelect"
        />
        <div class="drop-content">
          <div class="icon">📁</div>
          <p>点击或拖拽文件/文件夹到这里</p>
          <p class="hint">支持 .zip, .tar, .py, .cpp, .h, .md 等</p>
        </div>
      </div>

      <div v-if="fileList.length > 0" class="file-list">
        <h3>已选择 {{ fileList.length }} 个文件</h3>
        <ul>
          <li v-for="file in fileList.slice(0, 20)" :key="file.name">
            {{ file.name }} ({{ formatSize(file.size) }})
          </li>
          <li v-if="fileList.length > 20">... 还有 {{ fileList.length - 20 }} 个文件</li>
        </ul>
      </div>

      <div class="actions">
        <button
          class="btn-primary"
          :disabled="!store.canIndex"
          @click="startIndex"
        >
          🚀 开始建库
        </button>
        <button class="btn-secondary" @click="clearFiles">清空</button>
      </div>
    </section>

    <section v-if="store.isIndexing || store.indexProgress.total > 0" class="progress-section">
      <h3>⏳ 建库进度</h3>
      <div class="progress-bar">
        <div
          class="progress-fill"
          :style="{ width: progressPercent + '%' }"
        ></div>
      </div>
      <p class="progress-text">
        {{ store.indexProgress.current }} / {{ store.indexProgress.total }}
        - {{ store.indexProgress.message }}
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'
import { uploadFiles, buildIndex } from '@/api/client'

const store = useAppStore()
const isDragging = ref(false)
const fileList = ref<File[]>([])
const fileInput = ref<HTMLInputElement | null>(null)

const progressPercent = computed(() => {
  const { current, total } = store.indexProgress
  return total > 0 ? Math.round((current / total) * 100) : 0
})

function handleDrop(e: DragEvent) {
  isDragging.value = false
  const files = e.dataTransfer?.files
  if (files) {
    addFiles(files)
  }
}

function handleFileSelect(e: Event) {
  const files = (e.target as HTMLInputElement).files
  if (files) {
    addFiles(files)
  }
}

function addFiles(files: FileList) {
  for (let i = 0; i < files.length; i++) {
    fileList.value.push(files[i])
  }
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

async function startIndex() {
  if (fileList.value.length === 0) return

  try {
    store.setIndexing(true)
    
    // Upload files
    const dt = new DataTransfer()
    fileList.value.forEach(f => dt.items.add(f))
    const uploadRes = await uploadFiles(dt.files, 'web-upload')
    store.setProject(uploadRes.project_id, 'web-upload')

    // Build index with SSE
    const es = buildIndex({
      project_id: uploadRes.project_id,
      languages: 'python,c,cpp',
      chunk_backend: store.config.chunk_backend,
    })

    es.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.stage === 'done' || event.type === 'done') {
        store.setIndexed(true)
        store.setIndexing(false)
        store.setProgress(100, 100, '建库完成！')
        es.close()
        // Auto switch to query tab
        setTimeout(() => store.setActiveTab('query'), 800)
      } else if (data.current !== undefined) {
        store.setProgress(data.current, data.total, data.message)
      } else if (data.message) {
        store.setProgress(store.indexProgress.current, store.indexProgress.total, data.message)
      }
    }

    es.onerror = () => {
      store.setIndexing(false)
      store.setProgress(0, 0, '建库出错')
      es.close()
    }
  } catch (err) {
    store.setIndexing(false)
    alert('建库失败: ' + (err as Error).message)
  }
}

function clearFiles() {
  fileList.value = []
  store.setProgress(0, 0, '')
}
</script>

<style scoped>
.upload-view {
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

.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #f8fafc;
}

.drop-zone.dragging {
  border-color: #4f46e5;
  background: #eef2ff;
}

.drop-zone:hover {
  border-color: #4f46e5;
}

.icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.hint {
  color: #94a3b8;
  font-size: 13px;
  margin-top: 8px;
}

.file-list {
  margin-top: 16px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
}

.file-list h3 {
  font-size: 14px;
  margin-bottom: 8px;
  color: #475569;
}

.file-list ul {
  list-style: none;
  font-size: 13px;
  color: #64748b;
  max-height: 200px;
  overflow-y: auto;
}

.file-list li {
  padding: 4px 0;
  border-bottom: 1px solid #e2e8f0;
}

.actions {
  display: flex;
  gap: 12px;
  margin-top: 20px;
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
  transition: all 0.2s;
}

.btn-primary:hover:not(:disabled) {
  background: #4338ca;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  background: #94a3b8;
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

.progress-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.progress-bar {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  margin: 12px 0;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4f46e5, #7c3aed);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 13px;
  color: #64748b;
}
</style>
