"""
配置管理模組
管理所有系統配置參數
"""
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class Config:
    """系統配置類"""
    
    # Qdrant 配置
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "taiwan_law")
    
    # Embedding 服務（Ollama，由 OLLAMA_BASE_URL 的伺服器計算）
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "imac/zpoint_large_embedding_zh")

    # LLM 服務（OpenAI 相容端點：vLLM 或 Ollama 的 /v1 皆可）
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"{OLLAMA_BASE_URL}/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-26b")

    # 遠端服務逾時（秒）：伺服器卡住時不要讓查詢無限等待
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
    EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "120"))
    
    # RAG 配置
    TOP_K = int(os.getenv("TOP_K", "10"))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
    
    # Gradio 配置
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"

config = Config()

# 常用組合值
QDRANT_URL = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
QDRANT_COLLECTION_NAME = config.QDRANT_COLLECTION
