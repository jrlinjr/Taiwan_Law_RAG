"""
資料匯入模組

從 PDF 檔案載入文件（判決書、解釋函、法規等），
每頁一個 chunk 並保留頁碼，儲存至 Qdrant 向量資料庫。

主要功能：
- load_pdf_documents(): 從 PDF 檔案載入文件（每頁一個 chunk，保留頁碼）
- create_embeddings(): 初始化 Embedding 模型（由遠端 Ollama 計算）
- store_documents_in_qdrant(): 儲存文件至 Qdrant 向量資料庫（自動清除同來源舊資料）
- ingest_documents(): 主要的資料匯入函式（支援單一檔案或整個資料夾）

錯誤處理：
- DataIngestionError: 資料匯入錯誤基類
- QdrantConnectionError: Qdrant 連線錯誤
- PDFLoadError: PDF 載入錯誤
"""

import os # 查詢檔案路徑是否存在
import sys # sys.exit() 用於結束程式並返回狀態碼
from typing import List, Optional, Dict  # 用於類型註解

from pypdf import PdfReader # 用於讀取 PDF 文字
from qdrant_client import QdrantClient, models # Qdrant 連接客戶端與查詢條件
from langchain_core.documents import Document # LangChain 的 Document 類別
from langchain_ollama import OllamaEmbeddings # Embedding 模型（遠端 Ollama 計算）
from langchain_qdrant import QdrantVectorStore # Qdrant 向量資料庫

from config import (
    config,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
) # 配置物件與常數


class DataIngestionError(Exception):
    """資料匯入錯誤基類"""
    pass


class QdrantConnectionError(DataIngestionError):
    """Qdrant 連線錯誤"""
    pass


class PDFLoadError(DataIngestionError):
    """PDF 載入錯誤"""
    pass


class EmbeddingError(DataIngestionError):
    """Embedding 計算錯誤"""
    pass


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


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    從 PDF 檔案載入文件，每一頁切成一個 chunk 並保留頁碼。

    適合判決書、大法官解釋等沒有「第 X 條」結構的法律文件；
    頁碼會存進 metadata，供檢索結果標示引用來源。

    Args:
        pdf_path: PDF 檔案路徑

    Returns:
        List[Document]: 每頁一個 Document（已略過空白頁）

    Raises:
        PDFLoadError: 當 PDF 不存在、無法讀取或整份抽不到文字時
    """
    if not os.path.exists(pdf_path):
        raise PDFLoadError(
            f"找不到檔案：{pdf_path}\n"
            f"請檢查路徑是否正確。"
        )

    try:
        print(f"  載入 PDF 檔案... ({pdf_path})")
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise PDFLoadError(f"無法讀取 PDF 檔案：{str(e)}")

    # 以檔名（去除副檔名）作為文件名稱
    law_name = os.path.splitext(os.path.basename(pdf_path))[0]

    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # 略過空白頁（或無文字層的掃描頁）
        docs.append(Document(
            page_content=text,
            metadata={
                "law_name": law_name,
                "law_level": "",
                "law_category": "",
                "law_url": "",
                "modified_date": "",
                "article_no": "",
                "page": page_no,
                "source": os.path.basename(pdf_path),
                "source_type": "pdf",
            }
        ))

    if not docs:
        raise PDFLoadError(
            f"PDF 內沒有可抽取的文字：{pdf_path}\n"
            f"（可能是掃描影像 PDF，需要 OCR 才能讀取，本系統暫不支援）"
        )

    print(f"    ✓ {law_name}：{len(docs)} 頁")
    return docs


def create_embeddings() -> OllamaEmbeddings:
    """
    初始化 Embedding 模型（由 OLLAMA_BASE_URL 的伺服器計算）

    Returns:
        OllamaEmbeddings: LangChain Embeddings 實例

    Raises:
        EmbeddingError: 當模型初始化失敗時
    """
    print("\n初始化 Embedding 模型...")
    print(f"  model={config.EMBEDDING_MODEL} @ {config.OLLAMA_BASE_URL}")

    try:
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            client_kwargs={"timeout": config.OLLAMA_EMBED_TIMEOUT},
        )
        print(f"  ✓ 模型載入成功")
        return embeddings
    except Exception as e:
        raise EmbeddingError(
            f"Embedding 模型載入失敗（model={config.EMBEDDING_MODEL}）：{str(e)}"
        ) from e


def _remove_existing_sources(sources: List[str]) -> None:
    """
    刪除向量資料庫中來自相同來源檔案的舊資料。

    重複執行 ingest 時，避免同一份檔案的內容被重複累積，
    導致檢索結果被重複片段佔據。

    Args:
        sources: 來源檔案名稱清單（metadata.source 的值）
    """
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        return  # collection 尚未建立，無舊資料可清

    client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchAny(any=sources),
                )
            ])
        ),
    )
    print(f"  ✓ 已清除 {len(sources)} 個來源檔案的舊資料（避免重複匯入）")


def store_documents_in_qdrant(
    splits: List,
    embeddings,
    sources: Optional[List[str]] = None
) -> QdrantVectorStore:
    """
    使用 QdrantVectorStore.from_documents() 儲存文件至向量資料庫

    Args:
        splits: 切分後的文件片段列表
        embeddings: LangChain Embeddings 實例
        sources: 來源檔案名稱清單；提供時會先清除同來源的舊資料再寫入

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

    # from_documents() 內部會先計算 embedding 再寫入 Qdrant。
    # 先用一筆小文本驗證 embedding 服務可用，
    # 讓「embedding 失敗」與「Qdrant 失敗」能被分開回報，不會找錯方向。
    try:
        embeddings.embed_query("連線測試")
    except Exception as e:
        raise EmbeddingError(
            f"Embedding 計算失敗（model={config.EMBEDDING_MODEL}）：{str(e)}\n"
            f"請確認 Ollama 伺服器（{config.OLLAMA_BASE_URL}）正常運作且模型已下載。"
        )

    try:
        if sources:
            _remove_existing_sources(sources)

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


def _collect_source_files(source_path: str) -> List[str]:
    """
    將來源路徑展開為待匯入的 PDF 檔案清單。

    - 若為資料夾：遞迴掃描其中所有 .pdf 檔案（依路徑排序）
    - 若為單一檔案：回傳僅含該檔的清單

    Args:
        source_path: 檔案或資料夾路徑

    Returns:
        List[str]: 待匯入的檔案路徑清單
    """
    if os.path.isdir(source_path):
        files = []
        for root, _, names in os.walk(source_path):
            for name in names:
                if name.lower().endswith(".pdf"):
                    files.append(os.path.join(root, name))
        return sorted(files)
    return [source_path]


def ingest_documents(source_path: Optional[str] = None) -> Dict:
    """
    主要的資料匯入函式（PDF）。

    Args:
        source_path: 來源路徑，可為：
            - 單一 .pdf 檔案
            - 包含多個 .pdf 的資料夾（遞迴掃描）
            - None：使用預設的 data/uploads/ 資料夾

    Returns:
        Dict: 包含匯入統計資訊的字典

    Raises:
        DataIngestionError: 當匯入過程中發生錯誤時
    """
    # 使用預設路徑
    if source_path is None:
        # 從專案根目錄開始
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_path = os.path.join(project_root, "data", "uploads")

    print("=" * 60)
    print("台灣法律 RAG 系統 - 資料匯入 (PDF)")
    print("=" * 60)

    try:
        # 1. 展開來源檔案清單（支援單檔或整個資料夾）
        files = _collect_source_files(source_path)
        if not files:
            raise DataIngestionError(
                f"在 {source_path} 找不到可匯入的 .pdf 檔案"
            )

        print(f"\n[1/4] 載入來源檔案...（共 {len(files)} 個）")
        all_splits: List[Document] = []

        for file_path in files:
            print(f"\n  → {file_path}")
            if not file_path.lower().endswith(".pdf"):
                raise DataIngestionError(
                    f"不支援的檔案格式：{file_path}\n目前僅支援 .pdf"
                )
            all_splits.extend(load_pdf_documents(file_path))

        total_chunks = len(all_splits)
        if total_chunks == 0:
            raise DataIngestionError("沒有產生任何可儲存的文字片段，請檢查來源檔案內容")

        # 2. 切分結果統計
        print(f"\n[2/4] 切分完成，共 {total_chunks} 個文字片段")

        # 3. 初始化 Embedding
        print("\n[3/4] 初始化 Embedding 模型...")
        embeddings = create_embeddings()

        # 4. 儲存至 Qdrant（先清除同來源舊資料，避免重複匯入）
        print("\n[4/4] 儲存至向量資料庫...")
        source_names = [os.path.basename(f) for f in files]
        store_documents_in_qdrant(all_splits, embeddings, sources=source_names)

        # 顯示統計資訊
        print("\n" + "=" * 60)
        print("匯入完成！")
        print("=" * 60)
        print(f"來源檔案數：{len(files)}")
        print(f"文字片段數：{total_chunks}")
        print(f"向量資料庫：{QDRANT_COLLECTION_NAME}")
        print(f"Embedding 模型：{config.EMBEDDING_MODEL}")
        print("=" * 60)

        return {
            "success": True,
            "total_files": len(files),
            "total_chunks": total_chunks,
            "collection_name": QDRANT_COLLECTION_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
        }

    except QdrantConnectionError as e:
        print(f"\n❌ Qdrant 連線錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "qdrant"}
    except EmbeddingError as e:
        print(f"\n❌ Embedding 錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "embedding"}
    except PDFLoadError as e:
        print(f"\n❌ PDF 載入錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "pdf"}
    except DataIngestionError as e:
        print(f"\n❌ 資料匯入錯誤：\n{str(e)}")
        return {"success": False, "error": str(e), "error_type": "ingestion"}
    except Exception as e:
        print(f"\n❌ 未預期的錯誤：{str(e)}")
        return {"success": False, "error": str(e), "error_type": "unknown"}


if __name__ == "__main__":
    # 支援從命令列指定來源（檔案或資料夾）：
    #   python src/ingest.py                      # 預設掃描 data/uploads/
    #   python src/ingest.py data/某判決書.pdf     # 單一 PDF
    #   python src/ingest.py data/pdf/            # 整個資料夾（遞迴掃描 PDF）
    source = sys.argv[1] if len(sys.argv) > 1 else None
    result = ingest_documents(source)
    sys.exit(0 if result["success"] else 1)