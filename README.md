# 🏛️ 台灣法律智能問答系統

基於 RAG (Retrieval-Augmented Generation) 技術的台灣法律檢索與問答系統。

## 功能特色

- 🔍 **智能檢索**：使用向量資料庫快速檢索相關法律條文
- 🤖 **AI 生成回答**：結合 Ollama 本地 LLM 生成準確的法律解答
- 📚 **完整法律資料庫**：涵蓋台灣各類法律條文
- 🌐 **友善 Web 介面**：使用 Gradio 提供直覺的問答介面
- 🔒 **本地運行**：所有資料和運算都在本地，保護隱私
- ⚡ **Apple Silicon 優化**：針對 Mac M4 Pro 優化，使用 MPS 加速

## 系統架構

```
Taiwan_Law_RAG/
├── src/
│   ├── config.py      # 配置管理
│   ├── ingest.py      # 資料攝取
│   ├── rag.py         # RAG 核心邏輯
│   └── app.py         # Gradio Web UI
├── data/              # 法律資料
├── docs/              # 文檔
├── qdrant_storage/    # 向量資料庫儲存
└── requirements.txt   # Python 依賴
```

## 快速開始

### 1. 環境需求

- Python 3.10+
- Ollama (本地 LLM)
- Qdrant (向量資料庫)

### 2. 安裝依賴

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安裝套件
pip install -r requirements.txt
```

### 3. 啟動服務

```bash
# 啟動 Qdrant (使用 Docker)
docker-compose up -d

# 確認 Ollama 正在運行
ollama serve

# 下載模型 (首次使用)
ollama pull llama3.2:3b
```

### 4. 資料攝取

```bash
# 將法律資料匯入向量資料庫
python src/ingest.py data/chlaw.json/ChLaw.json
```

### 5. 啟動應用

```bash
# 啟動 Web UI
python src/app.py
```

開啟瀏覽器訪問 `http://localhost:7860`

## 配置說明

在專案根目錄建立 `.env` 檔案：

```env
# Qdrant 配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=taiwan_law

# Embedding 模型
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# RAG 參數
TOP_K=5
SCORE_THRESHOLD=0.5

# Gradio 配置
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False
```

## 使用範例

### 命令列查詢

```python
from src.rag import TaiwanLawRAG

rag = TaiwanLawRAG()
result = rag.query("什麼是詐欺罪？")
print(result['answer'])
```

### Web 介面

1. 啟動 `python src/app.py`
2. 在瀏覽器中輸入問題
3. 系統會顯示回答和相關法條來源

## 技術棧

- **LangChain**: RAG 框架
- **Ollama**: 本地 LLM 服務
- **Qdrant**: 向量資料庫
- **HuggingFace**: Embedding 模型
- **Gradio**: Web UI 框架

## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！
