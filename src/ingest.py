"""
資料匯入模組

支援從 JSON 格式載入台灣法規資料
使用 QdrantVectorStore.from_documents() 儲存至向量資料庫。

主要功能：
- load_json_documents(): 從 JSON 檔案載入法律資料
- split_documents(): 切分文件
- create_embeddings(): 使用 HuggingFace Embedding 模型將文字轉向量
- store_documents_in_qdrant(): 儲存文件至 Qdrant 向量資料庫
- ingest_documents(): 主要的資料匯入函式

錯誤處理：
- DataIngestionError: 資料匯入錯誤基類
- OllamaConnectionError: Ollama 連線錯誤
- QdrantConnectionError: Qdrant 連線錯誤
- JSONLoadError: JSON 載入錯誤
"""

import os # 查詢檔案路徑是否存在
import sys # sys.exit() 用於結束程式並返回狀態碼
import json # 用於載入 JSON 檔案
import requests # 用於 HTTP 請求
from typing import List, Dict, Optional  # 用於類型註解

from qdrant_client import QdrantClient # Qdrant 連接客戶端
from langchain_core.documents import Document # LangChain 的 Document 類別
from langchain_qdrant import QdrantVectorStore # Qdrant 向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings # HuggingFace Embedding 模型
from langchain_text_splitters import RecursiveCharacterTextSplitter # 用於切分文字的工具

from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    DATA_DIR,
) # 配置檔案中的常數


class DataIngestionError(Exception):
    """資料匯入錯誤基類"""
    pass


class OllamaConnectionError(DataIngestionError):
    """Ollama 連線錯誤"""
    pass


class QdrantConnectionError(DataIngestionError):
    """Qdrant 連線錯誤"""
    pass


class JSONLoadError(DataIngestionError):
    """JSON 載入錯誤"""
    pass


def check_ollama_connection() -> bool:
    """
    檢查 Ollama 服務是否可連接
    
    Returns:
        bool: 連接成功返回 True，否則返回 False
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
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
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return True
    except Exception:
        return False


def load_json_documents(json_path: str) -> List[Document]:
    """
    從 JSON 檔案載入台灣法規資料
    
    Args:
        json_path: JSON 檔案路徑
        
    Returns:
        List[Document]: 載入的文件列表，每個 Document 代表一個法律的完整資料
        
    Raises:
        JSONLoadError: 當 JSON 檔案不存在或無法讀取時
    """
    if not os.path.exists(json_path):
        raise JSONLoadError(
            f"找不到檔案：{json_path}\n"
            f"請檢查路徑是否正確。"
        )
    
    try:
        print(f"  載入 JSON 檔案... ({json_path})")
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        documents = []
        laws = data.get("Laws", [])
        
        for law in laws:
            law_name = law.get("LawName", "未知法律")
            law_level = law.get("LawLevel", "")         # 好像不太需要
            law_category = law.get("LawCategory", "")   # 意義不大
            articles = law.get("LawArticles", [])
            
            # 將 articles 轉換成可讀的文本格式
            page_content_lines = []
            for article in articles:
                article_type = article.get("ArticleType", "")
                article_no = article.get("ArticleNo", "").strip()
                article_content = article.get("ArticleContent", "").strip()
                
                if article_type == "C":
                    # 編/章標題
                    page_content_lines.append(f"\n{article_content}")
                elif article_type == "A" and article_content:
                    # 條文
                    if article_no:
                        page_content_lines.append(f"{article_no}\n{article_content}")
                    else:
                        page_content_lines.append(article_content)
            
            page_content = "\n".join(page_content_lines)
            
            # 將整個法律作為一個 Document，後續再切分
            doc = Document(
                page_content=page_content,
                metadata={
                    "law_name": law_name,
                    "law_level": law_level,
                    "law_category": law_category,
                    "law_url": law.get("LawURL", ""),
                    "modified_date": law.get("LawModifiedDate", ""),
                    "articles": articles  # 保留原始文章資料供 split_documents 使用
                }
            )
            documents.append(doc)
            print(f"    載入法律: {law_name} ({len(articles)} 條文)")
        
        print(f"    ✓ 成功載入 {len(documents)} 部法律")
        return documents
        
    except json.JSONDecodeError as e:
        raise JSONLoadError(f"JSON 格式錯誤：{str(e)}")
    except Exception as e:
        raise JSONLoadError(f"無法讀取 JSON 檔案：{str(e)}")


def split_documents(laws: List[Dict], chunk_size: int = 1000) -> List[Document]:
    """
    將每個條文轉換為 Document
    
    Args:
        laws: 法律資料列表
        chunk_size: 超過此大小的條文會進一步切分
        
    Returns:
        List[Document]: 切分後的 chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    
    print("\n切分文字...")
    splits = []
    
    for law in laws:
        law_name = law.get("LawName", "未知法律")
        law_level = law.get("LawLevel", "")
        law_category = law.get("LawCategory", "")
        law_url = law.get("LawURL", "")
        modified_date = law.get("LawModifiedDate", "")
        articles = law.get("LawArticles", [])
        
        for article in articles:
            article_type = article.get("ArticleType", "")
            article_no = article.get("ArticleNo", "").strip()
            article_content = article.get("ArticleContent", "").strip()
            
            # 只處理條文（ArticleType: "A"）
            if article_type != "A" or not article_content:
                continue
            
            # 建立 chunk 內容
            if article_no:
                chunk_text = f"{article_no}\n{article_content}"
            else:
                chunk_text = article_content
            
            # 如果超過 chunk_size，進一步切分
            if len(chunk_text) > chunk_size:
                sub_chunks = splitter.create_documents([chunk_text])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update({
                        "law_name": law_name,
                        "law_level": law_level,
                        "law_category": law_category,
                        "law_url": law_url,
                        "modified_date": modified_date,
                        "article_no": article_no,
                    })
                    splits.append(sub_chunk)
            else:
                splits.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "law_name": law_name,
                        "law_level": law_level,
                        "law_category": law_category,
                        "law_url": law_url,
                        "modified_date": modified_date,
                        "article_no": article_no,
                    }
                ))
    
    print(f"  ✓ 產生 {len(splits)} 個文字片段")
    return splits


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    初始化 HuggingFace Embedding 模型
    
    Returns:
        HuggingFaceEmbeddings: HuggingFace embeddings 實例
        
    Raises:
        Exception: 當模型載入失敗時
    """
    print("\n初始化 HuggingFace Embedding 模型...")
    print(f"  正在載入模型：{EMBEDDING_MODEL}...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"  ✓ 模型載入成功")
        return embeddings
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            raise Exception(
                f"模型 {EMBEDDING_MODEL} 未找到或無法下載\n"
                f"錯誤：{error_msg}\n"
                f"請確保網路連接正常，或使用其他可用的 HuggingFace 模型"
            )
        raise Exception(f"模型載入失敗：{error_msg}")


def store_documents_in_qdrant(
    splits: List,
    embeddings: HuggingFaceEmbeddings
) -> QdrantVectorStore:
    """
    使用 QdrantVectorStore.from_documents() 儲存文件至向量資料庫
    
    Args:
        splits: 切分後的文件片段列表
        embeddings: HuggingFaceEmbeddings 實例
        
    Returns:
        QdrantVectorStore: Qdrant vector store 實例
        
    Raises:
        QdrantConnectionError: 當無法連接 Qdrant 時
    """
    print("\n儲存至向量資料庫...")
    
    if not check_qdrant_connection():
        raise QdrantConnectionError(
            f"無法連接 Qdrant 服務（{QDRANT_URL}）\n"
            f"請確認 Qdrant 容器是否運行：docker-compose up -d\n"
            f"或檢查 QDRANT_URL 環境變數設定是否正確。"
        )
    
    try:
        vectorstore = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME,
            prefer_grpc=False
        )
        print(f"  ✓ 成功儲存 {len(splits)} 個文字片段至 Qdrant")
        return vectorstore
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise QdrantConnectionError(
                f"Qdrant 連線失敗：{str(e)}\n"
                f"請確認 Qdrant 服務是否正常運行。"
            )
        raise QdrantConnectionError(f"儲存至 Qdrant 失敗：{str(e)}")


def ingest_documents(
    json_path: Optional[str] = None
) -> Dict:
    """
    主要的資料匯入函式（JSON 版本）
    
    Args:
        json_path: JSON 檔案路徑（若為 None，使用預設路徑）
        
    Returns:
        Dict: 包含匯入統計資訊的字典
        
    Raises:
        DataIngestionError: 當匯入過程中發生錯誤時
    """
    # 使用預設路徑
    if json_path is None:
        # 從專案根目錄開始
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "data", "chlaw.json", "ChLaw.json")
    
    print("=" * 60)
    print("台灣法律 RAG 系統 - 資料匯入 (JSON)")
    print("=" * 60)
    
    try:
        # 1. 載入 JSON 檔案
        print("\n[1/4] 載入 JSON 檔案...")
        documents = load_json_documents(json_path)
        total_laws = len(documents)
        
        # 提取法律資料
        laws = []
        for doc in documents:
            law_data = {
                "LawName": doc.metadata.get("law_name"),
                "LawLevel": doc.metadata.get("law_level"),
                "LawCategory": doc.metadata.get("law_category"),
                "LawURL": doc.metadata.get("law_url"),
                "LawModifiedDate": doc.metadata.get("modified_date"),
                "LawArticles": doc.metadata.get("articles", [])
            }
            laws.append(law_data)
        
        # 2. 切分文字
        print("\n[2/4] 切分文字...")
        splits = split_documents(laws)
        total_chunks = len(splits)
        
        # 3. 初始化 Embedding
        print("\n[3/4] 初始化 Embedding 模型...")
        embeddings = create_embeddings()
        
        # 4. 儲存至 Qdrant
        print("\n[4/4] 儲存至向量資料庫...")
        vectorstore = store_documents_in_qdrant(splits, embeddings)
        
        # 顯示統計資訊
        print("\n" + "=" * 60)
        print("匯入完成！")
        print("=" * 60)
        print(f"總法律數：{total_laws}")
        print(f"文字片段數：{total_chunks}")
        print(f"向量資料庫：{QDRANT_COLLECTION_NAME}")
        print(f"Embedding 模型：{EMBEDDING_MODEL}")
        print("=" * 60)
        
        return {
            "success": True,
            "total_laws": total_laws,
            "total_chunks": total_chunks,
            "collection_name": QDRANT_COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }
        
    except OllamaConnectionError as e:
        print(f"\n❌ Ollama 連線錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "ollama"}
    except QdrantConnectionError as e:
        print(f"\n❌ Qdrant 連線錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "qdrant"}
    except JSONLoadError as e:
        print(f"\n❌ JSON 載入錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "json"}
    except Exception as e:
        print(f"\n❌ 未預期的錯誤：{str(e)}")
        return {"success": False, "error": str(e), "error_type": "unknown"}


if __name__ == "__main__":
    result = ingest_documents()
    sys.exit(0 if result["success"] else 1)