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
from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

from config import config


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


def create_rag_chain() -> Dict:
    """
    建立 RAG 查詢鏈
    
    這是系統的核心函式，負責：
    1. 檢查服務連接
    2. 初始化 Embedding 模型
    3. 連接向量資料庫
    4. 建立 Retriever
    5. 初始化 LLM
    6. 組合 RAG Chain
    
    Returns:
        Dict: 包含以下鍵值：
            - chain: LangChain RAG Chain
            - retriever: 向量檢索器
            - vector_store: Qdrant 向量資料庫
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
        
        # 2. 初始化 Embeddings
        print(f"載入 Embedding 模型: {config.EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding 模型載入成功")
        
        # 3. 連接向量資料庫
        print(f"連接 Qdrant Collection: {config.QDRANT_COLLECTION}")
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=config.QDRANT_COLLECTION,
            url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
        )
        print("✓ 向量資料庫連接成功")
        
        # 4. 建立 Retriever
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": config.TOP_K,
                "score_threshold": config.SCORE_THRESHOLD
            }
        )
        
        # 5. 初始化 LLM
        print(f"初始化 LLM: {config.OLLAMA_MODEL}")
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0.7
        )
        print("✓ LLM 初始化成功")
        
        # 6. 建立 Prompt Template
        template = """你是一位精通中華民國台灣法律的法律顧問，用簡潔專業的方式為不懂法律的人解釋法律。

【相關法律條文】
{context}

【使用者問題】
{question}

【回答方式 - 先重點後詳解】

**【重點摘要】**（最先呈現）
- 依據：[法律名稱第X條]，罪名/法律效果：[具體罪名或法律後果]
- 簡述：[一句話說明]

**一、法律規範**
- 引用相關法律名稱及條文編號
- 簡述法條內容（1-2句）
- 說明主要構成要件（用簡單列點，最多3-4項）

**二、法律解釋**
- 用白話文解釋法條的意思
- 說明這個法律在什麼情況下適用
- 補充實務見解或常見案例

**三、白話舉例（根據使用者問題）**
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
        
        # 7. 格式化文件函數
        def format_docs(docs):
            """
            格式化檢索到的文件
            
            Args:
                docs: LangChain Document 物件列表
                
            Returns:
                str: 格式化後的文件字串
            """
            formatted = []
            for doc in docs:
                metadata = doc.metadata
                law_name = metadata.get('law_name', '未知法律')
                article_no = metadata.get('article_no', '未知條文')
                content = doc.page_content
                formatted.append(f"【{law_name} 第{article_no}條】\n{content}")
            return "\n\n".join(formatted)
        
        # 8. 建立 RAG Chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✓ RAG 系統初始化完成\n")
        
        return {
            "chain": chain,
            "retriever": retriever,
            "vector_store": vector_store,
            "embeddings": embeddings
        }
        
    except (OllamaConnectionError, QdrantConnectionError) as e:
        raise e
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
        rag_chain_dict: 包含 chain 和 retriever 的字典
        
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
        retriever = rag_chain_dict["retriever"]
        
        # 檢索相關文件
        docs = retriever.invoke(question)
        
        # 生成回答
        answer = chain.invoke(question)
        
        # 整理來源
        sources = []
        for doc in docs:
            metadata = doc.metadata
            sources.append({
                'law_name': metadata.get('law_name', ''),
                'article_no': metadata.get('article_no', ''),
                'content': doc.page_content,
                'score': 0.0,
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
