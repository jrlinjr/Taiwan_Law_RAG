# 🏛️ 中華民國法律智能問答系統

基於 RAG (Retrieval-Augmented Generation) 技術的台灣法律檢索與問答系統。

使用專業法律人的三段論模式（大前提、小前提、結論），為不懂法律的民眾提供清晰易懂的法律解答。

## 功能特色

- 🔍 **智能檢索**：使用向量資料庫快速檢索相關法律條文
- 🤖 **AI 生成回答**：結合 Ollama 本地 LLM 生成準確的法律解答
- 📚 **完整法律資料庫**：涵蓋台灣各類法律條文
- 🌐 **友善 Web 介面**：使用 Gradio 提供直覺的問答介面
- 🔒 **本地運行**：所有資料和運算都在本地，保護隱私
- ⚡ **Apple Silicon 優化**：針對 Mac M4 Pro 優化，使用 MPS 加速
- 📖 **專業法律解釋**：三段論模式 + 白話舉例，讓民眾理解法律

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
ollama pull gpt-oss:20b  # 在遠端伺服器上執行
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

# Embedding 模型 (Mac M4 Pro 優化)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=mps

# Ollama 配置 (遠端伺服器)
OLLAMA_BASE_URL=http://10.0.0.209:11434
OLLAMA_MODEL=gpt-oss:20b

# RAG 參數
TOP_K=10
SCORE_THRESHOLD=0.5

# Gradio 配置
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False
```

### 配置說明

- **EMBEDDING_MODEL**: 使用 `BAAI/bge-m3` (1024 維多語言模型) 以獲得最佳效果
- **EMBEDDING_DEVICE**: Mac M4 Pro 使用 `mps` 以利用 Apple Silicon GPU 加速
- **OLLAMA_BASE_URL**: 遠端 Ollama 伺服器地址 (10.0.0.209)
- **OLLAMA_MODEL**: 使用 `gpt-oss:20b` 提供高品質的法律解答
- **TOP_K**: 檢索前 10 個最相關的法條

## 使用範例

### Web 介面（推薦）

```bash
# 啟動應用
python src/app.py

# 在瀏覽器中訪問
# http://localhost:7860
```

### 命令列查詢

```python
from src.rag import create_rag_chain, query

# 建立 RAG 系統
rag_chain_dict = create_rag_chain()

# 執行查詢
result = query("什麼是詐欺罪？", rag_chain_dict)

# 顯示結果
print("回答:")
print(result['answer'])
print("\n來源法條:")
for source in result['sources']:
    print(f"- {source['law_name']} 第{source['article_no']}條")
```

### 回答格式

系統使用三段論模式回答：

```
【重點摘要】
- 依據：刑法第 339 條
- 罪名：詐欺罪
- 簡述：以詐術使人交付財物

【一、法律規範】
- 法條內容和構成要件

【二、法律解釋】
- 白話文解釋和適用情況

【三、白話舉例】
- 生活化的具體例子
- 說明法律後果
```

## 技術棧

- **LangChain**: RAG 框架
- **Ollama**: 遠端 LLM 服務 (gpt-oss:20b @ 10.0.0.209)
- **Qdrant**: 向量資料庫 (本地)
- **HuggingFace**: Embedding 模型 (BAAI/bge-m3)
- **Gradio**: Web UI 框架

## 文檔

- [RAG 模組文檔](docs/RAG_MODULE.md) - RAG 系統詳細說明
- [資料攝取文檔](docs/INGEST_MODULE.md) - 資料匯入流程
- [Web UI 文檔](docs/APP_MODULE.md) - 應用程式說明
- [Mac M4 Pro 優化指南](docs/MAC_M4_OPTIMIZATION.md) - Apple Silicon 優化

## 常見問題

### Q: 如何更改 LLM 模型？

編輯 `.env` 檔案（需在遠端伺服器上下載模型）：
```env
OLLAMA_MODEL=qwen2.5:14b
```

### Q: 如何提升回答品質？

1. 增加檢索數量：`TOP_K=15`
2. 使用更大的 Embedding 模型：`BAAI/bge-large-zh-v1.5`
3. 確保遠端 Ollama 伺服器正常運行

### Q: 如何加快查詢速度？

1. 減少檢索數量：`TOP_K=5`
2. 提高相似度門檻：`SCORE_THRESHOLD=0.7`
3. 檢查網路連線到遠端 Ollama 伺服器

## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！

---

**最後更新**: 2025 年 12 月
