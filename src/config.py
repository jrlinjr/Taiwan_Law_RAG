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
    
    # Embedding 模型配置
    # provider 為 "ollama" 時由 OLLAMA_BASE_URL 的伺服器計算（與 sysbrain 一致），
    # 為 "huggingface" 時在本機計算（EMBEDDING_DEVICE 僅此模式使用）
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "imac/zpoint_large_embedding_zh")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "mps")  # 使用 Apple Silicon GPU
    
    # Ollama 配置（實際伺服器位址請在 .env 設定，勿寫死於程式碼）
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    # 遠端呼叫逾時（秒）：伺服器卡住時不要讓查詢無限等待
    OLLAMA_LLM_TIMEOUT = int(os.getenv("OLLAMA_LLM_TIMEOUT", "300"))
    OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "120"))
    
    # RAG 配置
    TOP_K = int(os.getenv("TOP_K", "10"))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
    
    # Gradio 配置
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"

config = Config()

# 向後相容：提供舊的變數名稱
OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DEVICE = config.EMBEDDING_DEVICE
QDRANT_URL = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
QDRANT_COLLECTION_NAME = config.QDRANT_COLLECTION
DATA_DIR = "data"
