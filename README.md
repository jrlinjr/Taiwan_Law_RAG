# 🏛️ 中華民國法律智能問答系統

基於 RAG (Retrieval-Augmented Generation) 技術的台灣法律檢索與問答系統。

上傳法律 PDF（判決書、解釋函、法規等），系統會檢索相關內容，再由 LLM 生成
「法律規範 → 法律解釋 → 白話舉例」三段式的解答，讓不懂法律的民眾也能理解。

## 功能特色

- 🔍 **語意檢索**：以向量資料庫檢索與問題最相關的法律內容
- 🤖 **AI 生成回答**：以 LLM 生成準確、易懂的法律解答
- 📄 **PDF 匯入**：上傳 PDF 即可加入知識庫（按頁切分、保留頁碼當引用來源）
- 🌐 **友善 Web 介面**：使用 Gradio，可在頁面上直接上傳 PDF 與提問
- 📚 **來源可追溯**：每筆回答附上參考的法條/頁碼與相似度分數

## 系統架構

運算可分散到不同服務，各司其職：

```
使用者瀏覽器
     │
     ▼
  Gradio App ──┬──► Qdrant 向量資料庫（檢索）
               ├──► Ollama（Embedding，OLLAMA_BASE_URL）
               └──► OpenAI 相容端點（LLM 生成，LLM_BASE_URL：vLLM 或 Ollama /v1）
```

```
Taiwan_Law_RAG/
├── src/
│   ├── config.py      # 配置管理（讀取 .env）
│   ├── ingest.py      # PDF 匯入與 Embedding
│   ├── rag.py         # RAG 核心：檢索 + LLM 生成
│   └── app.py         # Gradio Web UI
├── data/uploads/      # 上傳的 PDF 原始檔
├── qdrant_storage/    # 向量資料庫本地儲存
├── docker-compose.yaml
└── requirements.txt
```

## 快速開始

### 1. 環境需求

- Python 3.10+
- Qdrant（向量資料庫，可用 Docker 啟動）
- 一個 Ollama 伺服器（提供 Embedding 模型）
- 一個 OpenAI 相容的 LLM 端點（vLLM 或 Ollama 的 `/v1`）

### 2. 安裝依賴

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
pip install -r requirements.txt
```

### 3. 設定 `.env`

在專案根目錄建立 `.env`（參考下方「配置說明」）。

### 4. 啟動 Qdrant

```bash
docker compose up -d
```

### 5. 匯入資料

支援 PDF（按頁切分、保留頁碼）。也可以直接在 Web UI 上傳，原始檔會存到 `data/uploads/`。

```bash
python src/ingest.py                    # 匯入 data/uploads/ 內所有 PDF
python src/ingest.py data/某判決書.pdf   # 單一 PDF
python src/ingest.py data/pdf/          # 整個資料夾（遞迴掃描 .pdf）
```

重複匯入同一份檔案會自動清除舊資料後再寫入，不會產生重複內容。

### 6. 啟動應用

```bash
python src/app.py
```

開啟瀏覽器訪問 `http://localhost:7860`。

## 配置說明

`.env` 範例：

```env
# Qdrant 向量資料庫
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=taiwan_law

# Embedding 服務（Ollama）
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=imac/zpoint_large_embedding_zh

# LLM 服務（OpenAI 相容端點：vLLM 或 Ollama 的 /v1）
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma-4-26b

# 遠端服務逾時（秒）
LLM_TIMEOUT=300
EMBED_TIMEOUT=120

# RAG 參數
TOP_K=8
SCORE_THRESHOLD=0.5

# Gradio
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False
```

| 設定 | 說明 |
|------|------|
| `OLLAMA_BASE_URL` / `EMBEDDING_MODEL` | Embedding 服務位址與模型（Ollama） |
| `LLM_BASE_URL` / `LLM_MODEL` | LLM 服務位址與模型；OpenAI 相容，vLLM 或 Ollama `/v1` 皆可 |
| `TOP_K` | 每次檢索取前 K 個片段（越大回答越完整，但 LLM 生成越慢） |
| `SCORE_THRESHOLD` | 相似度門檻，低於此值的片段不納入 context |

> Embedding 與 LLM 可指向不同主機。例如：LLM 用一台有 GPU 的機器跑 vLLM，
> Embedding 指向另一台有 Ollama 的機器，兩者各自獨立設定即可。

## 使用範例

### 命令列查詢

```python
import sys; sys.path.insert(0, "src")
from rag import create_rag_chain, query

rag_chain_dict = create_rag_chain()
result = query("什麼是詐欺罪？", rag_chain_dict)

print(result["answer"])
for s in result["sources"]:
    label = s["article_no"] or f"第{s['page']}頁"
    print(f"- {s['law_name']} {label}（相似度 {s['score']}）")
```

### 回答格式

```
依據刑法第 339 條之規定，行為人之詐欺行為成立詐欺罪。

**法律規範**
- 法條內容和主要構成要件

**法律解釋**
- 白話文解釋和適用情況

**白話舉例**
- 生活化的具體例子
- 說明法律後果（罰款、監禁、民事賠償等）
```

## 技術棧

- **LangChain**：RAG 框架
- **Qdrant**：向量資料庫
- **Ollama**：Embedding 服務（`imac/zpoint_large_embedding_zh`）
- **vLLM / Ollama**：LLM 生成（OpenAI 相容端點）
- **pypdf**：PDF 文字抽取
- **Gradio**：Web UI

## 常見問題

### Q: 如何更換 LLM 模型？

編輯 `.env` 的 `LLM_MODEL`（並確認該模型存在於 `LLM_BASE_URL` 的服務上）。

### Q: 如何提升回答品質 / 加快速度？

- 提升完整度：調高 `TOP_K`
- 加快速度：調低 `TOP_K`、或提高 `SCORE_THRESHOLD`
- 更換更適合的 LLM 或 Embedding 模型

### Q: 掃描影像 PDF 匯入失敗？

沒有文字層的掃描 PDF 需要 OCR 才能抽取文字，本系統暫不支援。

## 授權

MIT License
