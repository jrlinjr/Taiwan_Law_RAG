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
    
    # Embedding 模型配置 (Mac M4 Pro 優化)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "mps")  # 使用 Apple Silicon GPU
    
    # Ollama 配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.0.0.209:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    
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
