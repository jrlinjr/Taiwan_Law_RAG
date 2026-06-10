"""
RAG (Retrieval-Augmented Generation) 模組
負責檢索相關法律條文並生成回答

主要功能：
- check_ollama_connection(): 檢查 Ollama 連接
- check_qdrant_connection(): 檢查 Qdrant 連接
- create_rag_chain(): 建立 RAG 查詢鏈
- query(): 執行查詢

錯誤處理：
- RAGError: RAG 相關錯誤基類
- OllamaConnectionError: Ollama 連線錯誤
- QdrantConnectionError: Qdrant 連線錯誤
"""

import requests
from typing import List, Dict

from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

from config import config
from ingest import create_embeddings


class RAGError(Exception):
    """RAG 相關錯誤基類"""
    pass


class OllamaConnectionError(RAGError):
    """Ollama 連線錯誤"""
    pass


class QdrantConnectionError(RAGError):
    """Qdrant 連線錯誤"""
    pass


def check_ollama_connection() -> bool:
    """
    檢查 Ollama 服務是否可連接
    
    Returns:
        bool: 連接成功返回 True，否則返回 False
    """
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> List[str]:
    """
    取得 Ollama 伺服器上可用的模型名稱清單

    Returns:
        List[str]: 模型名稱列表（含 tag，如 "gemma4:12b"）；查詢失敗時回傳空列表
    """
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return []
        return [m.get("name", "") for m in response.json().get("models", [])]
    except Exception:
        return []


def check_qdrant_connection() -> bool:
    """
    檢查 Qdrant 服務是否可連接
    
    Returns:
        bool: 連接成功返回 True，否則返回 False
    """
    try:
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        client.get_collections()
        return True
    except Exception:
        return False


def format_docs(docs: List) -> str:
    """
    將檢索到的文件格式化為 prompt 的 context 字串

    Args:
        docs: LangChain Document 物件列表

    Returns:
        str: 格式化後的文件字串
    """
    formatted = []
    for doc in docs:
        metadata = doc.metadata
        law_name = metadata.get('law_name', '未知法律')
        article_no = metadata.get('article_no', '')
        page = metadata.get('page')
        # article_no 在攝取時已是完整「第 X 條」；PDF 來源則改用頁碼標示
        if article_no:
            label = f"{law_name} {article_no}"
        elif page:
            label = f"{law_name} 第{page}頁"
        else:
            label = law_name
        formatted.append(f"【{label}】\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_chain() -> Dict:
    """
    建立 RAG 查詢鏈

    這是系統的核心函式，負責：
    1. 檢查服務連接
    2. 初始化 Embedding 模型
    3. 連接向量資料庫
    4. 初始化 LLM
    5. 組合 RAG Chain

    Returns:
        Dict: 包含以下鍵值：
            - chain: LangChain Chain（輸入 {"context", "question"}）
            - vector_store: Qdrant 向量資料庫（檢索用）
            - embeddings: Embedding 模型

    Raises:
        OllamaConnectionError: 無法連接 Ollama
        QdrantConnectionError: 無法連接 Qdrant
        RAGError: 其他初始化錯誤
    """
    print("\n初始化 RAG 系統...")
    
    try:
        # 1. 檢查服務連接
        print("檢查服務連接...")
        if not check_ollama_connection():
            raise OllamaConnectionError(
                f"無法連接 Ollama 服務（{config.OLLAMA_BASE_URL}）\n"
                f"請確認 Ollama 是否正在運行"
            )
        
        if not check_qdrant_connection():
            raise QdrantConnectionError(
                f"無法連接 Qdrant 服務（{config.QDRANT_HOST}:{config.QDRANT_PORT}）\n"
                f"請確認 Qdrant 是否正在運行"
            )
        
        print("✓ 服務連接正常")

        # 1.5 確認所需模型存在於伺服器
        # （啟動時就攔截「模型未下載或損壞」，不要等到第一次查詢才發現）
        available = get_available_models()
        required = [config.OLLAMA_MODEL, config.EMBEDDING_MODEL]
        # /api/tags 回傳的名稱一律帶 tag（未指定時為 :latest）
        missing = [
            m for m in required
            if m not in available and f"{m}:latest" not in available
        ]
        if missing:
            raise OllamaConnectionError(
                f"Ollama 伺服器（{config.OLLAMA_BASE_URL}）上找不到模型："
                f"{', '.join(missing)}\n"
                f"請先在伺服器上執行：ollama pull <模型名稱>\n"
                f"或修改 .env 改用伺服器上已有的模型。"
            )
        print(f"✓ 模型確認存在：{', '.join(required)}")

        # 2. 初始化 Embeddings（由遠端 Ollama 計算）
        embeddings = create_embeddings()
        
        # 3. 連接向量資料庫
        print(f"連接 Qdrant Collection: {config.QDRANT_COLLECTION}")
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        if not client.collection_exists(config.QDRANT_COLLECTION):
            raise RAGError(
                f"向量資料庫中沒有 Collection「{config.QDRANT_COLLECTION}」，"
                f"表示尚未匯入任何資料。\n"
                f"請先匯入資料（擇一）：\n"
                f"  1. 在 Web 介面展開「📥 匯入 PDF 到知識庫」上傳 PDF\n"
                f"  2. 命令列匯入：python src/ingest.py [PDF 或資料夾]"
            )
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=config.QDRANT_COLLECTION,
            url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
        )
        print("✓ 向量資料庫連接成功")
        
        # 4. 初始化 LLM
        # temperature 取低值：法律問答需要穩定、可重現、貼近法條原文的回答
        print(f"初始化 LLM: {config.OLLAMA_MODEL}")
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0.2,
            client_kwargs={"timeout": config.OLLAMA_LLM_TIMEOUT},
        )
        print("✓ LLM 初始化成功")

        # 5. 建立 Prompt Template
        template = """你是一位精通中華民國台灣法律的法律顧問，用簡潔專業的方式為不懂法律的人解釋法律。

【相關法律條文】
{context}

【使用者問題】
{question}

【回答方式 - 專業法律分析】

請直接開始回答，格式如下：

依據[法律名稱第X條]之規定，[行為人]之[犯罪行為]行為成立[罪名]。

**法律規範**
- 引用相關法律名稱及條文編號
- 完整陳述法條的規範內容
- 說明主要構成要件（用簡單列點，最多3-4項）

**法律解釋**
- 用白話文解釋法條的意思
- 說明這個法律在什麼情況下適用
- 補充實務見解或常見案例

**白話舉例**
- 根據使用者提出的問題情境，舉一個相關的具體例子
- 說明在這個例子中，法律如何應用
- 說明這個行為會帶給民眾什麼法律後果（如罰款、監禁、民事賠償等）
- 讓人能夠理解法律對日常生活的實際影響

【重點提醒】
- 回答要簡潔，避免冗長
- 使用繁體中文及中華民國法律用語
- 舉例要貼近使用者的問題，讓人感受到法律的實際意義和後果
- 本回答僅供初步了解，實際個案請諮詢執業律師

請開始回答："""
        
        prompt = ChatPromptTemplate.from_template(template)

        # 6. 建立 Chain
        # 檢索交由 query() 以 similarity_search_with_score() 執行一次完成，
        # chain 只負責「context + question → 回答」，避免重複檢索
        chain = prompt | llm | StrOutputParser()

        print("✓ RAG 系統初始化完成\n")

        return {
            "chain": chain,
            "vector_store": vector_store,
            "embeddings": embeddings
        }
        
    except RAGError:
        # 含 OllamaConnectionError、QdrantConnectionError，訊息已經友善，直接拋出
        raise
    except Exception as e:
        raise RAGError(f"初始化 RAG 系統失敗: {str(e)}")


def query(question: str, rag_chain_dict: Dict) -> Dict:
    """
    執行查詢
    
    這是系統的查詢入口，負責：
    1. 驗證輸入
    2. 檢索相關文件
    3. 生成回答
    4. 整理來源
    
    Args:
        question: 使用者問題
        rag_chain_dict: 包含 chain 和 vector_store 的字典
        
    Returns:
        Dict: 包含以下鍵值：
            - answer: LLM 生成的回答
            - sources: 相關法條來源列表
            - success: 查詢是否成功
            
    Raises:
        RAGError: 當查詢失敗時
    """
    if not question or not question.strip():
        return {
            'answer': "請輸入問題",
            'sources': [],
            'success': False
        }
    
    try:
        chain = rag_chain_dict["chain"]
        vector_store = rag_chain_dict["vector_store"]

        # 檢索相關文件（一次完成，同時取得相似度分數並套用門檻）
        docs_with_scores = vector_store.similarity_search_with_score(
            question,
            k=config.TOP_K,
            score_threshold=config.SCORE_THRESHOLD,
        )

        # 生成回答（將檢索結果直接作為 context，不再重複檢索）
        docs = [doc for doc, _ in docs_with_scores]
        context = format_docs(docs) if docs else "（未檢索到相關法條）"
        answer = chain.invoke({"context": context, "question": question})

        # 整理來源（含真實相似度分數）
        sources = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            sources.append({
                'law_name': metadata.get('law_name', ''),
                'article_no': metadata.get('article_no', ''),
                'page': metadata.get('page'),
                'content': doc.page_content,
                'score': round(float(score), 4),
                'url': metadata.get('law_url', '')
            })

        return {
            'answer': answer,
            'sources': sources,
            'success': True
        }
        
    except Exception as e:
        return {
            'answer': f"查詢時發生錯誤: {str(e)}",
            'sources': [],
            'success': False
        }


if __name__ == "__main__":
    # 測試
    try:
        print("=" * 60)
        print("RAG 系統測試")
        print("=" * 60)
        
        rag_chain_dict = create_rag_chain()
        result = query("什麼是詐欺罪？", rag_chain_dict)
        
        print("\n" + "=" * 60)
        print("回答:")
        print("=" * 60)
        print(result['answer'])
        
        print("\n" + "=" * 60)
        print(f"來源數量: {len(result['sources'])}")
        print("=" * 60)
        
    except RAGError as e:
        print(f"❌ 錯誤: {e}")
