"""
Gradio Web UI 應用程式
提供友善的網頁介面進行法律問答

主要功能：
- initialize_rag_chain(): 初始化 RAG 系統
- answer_question(): 處理使用者問題
- create_web_ui(): 建立 Gradio UI
- main(): 啟動應用程式
"""

import os
import shutil
import gradio as gr
from typing import Tuple

from rag import create_rag_chain, query, RAGError
from ingest import (
    load_pdf_documents,
    create_embeddings,
    store_documents_in_qdrant,
    DataIngestionError,
)
from config import config


# 上傳的 PDF 原始檔保存位置（專案根目錄下 data/uploads/）
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "uploads"
)

# 全域快取：RAG Chain 與 Embedding 模型（避免重複載入）
_rag_chain_dict = None
_embeddings = None


def initialize_rag_chain():
    """
    初始化 RAG 系統
    
    使用全域快取避免重複初始化，提升效能。

    Returns:
        Dict: 包含 chain 和 vector_store 的字典
    """
    global _rag_chain_dict
    
    if _rag_chain_dict is None:
        _rag_chain_dict = create_rag_chain()

    return _rag_chain_dict


def get_embeddings():
    """
    取得 Embedding 模型（全域快取）

    優先重用已初始化 RAG 系統內的模型；若 RAG 尚未初始化
    （例如知識庫還是空的、第一次就先上傳檔案），則獨立載入，
    讓「先上傳、後問答」的順序也能運作。

    Returns:
        Embeddings: LangChain Embeddings 實例（依 EMBEDDING_PROVIDER 而定）
    """
    global _embeddings

    if _embeddings is None:
        if _rag_chain_dict is not None:
            _embeddings = _rag_chain_dict["embeddings"]
        else:
            _embeddings = create_embeddings()

    return _embeddings


def upload_and_ingest(file_paths) -> str:
    """
    處理使用者上傳的 PDF，匯入向量資料庫

    原始檔會複製保存到 data/uploads/，再從該位置匯入，
    因此即使 Gradio 暫存被清除，原始 PDF 仍保留在專案內。

    Args:
        file_paths: Gradio File 元件回傳的檔案路徑列表

    Returns:
        str: 匯入結果訊息
    """
    if not file_paths:
        return "請先選擇要匯入的 PDF 檔案"

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    results = []
    for path in file_paths:
        name = os.path.basename(path)
        try:
            # 保存原始檔（已存在同名檔案則覆蓋，與向量庫的去重行為一致）
            dest = os.path.join(UPLOAD_DIR, name)
            if os.path.abspath(path) != os.path.abspath(dest):
                shutil.copy(path, dest)

            splits = load_pdf_documents(dest)
            store_documents_in_qdrant(splits, get_embeddings(), sources=[name])
            results.append(f"✓ {name}：成功匯入 {len(splits)} 個片段（原始檔已保存至 data/uploads/）")
        except DataIngestionError as e:
            results.append(f"❌ {name}：{str(e)}")
        except Exception as e:
            results.append(f"❌ {name}：發生錯誤 {str(e)}")

    results.append("\n匯入的內容可立即在上方問答中被檢索。")
    return "\n".join(results)


def answer_question(question: str) -> Tuple[str, str]:
    """
    處理使用者問題
    
    Args:
        question: 使用者輸入的問題
        
    Returns:
        Tuple[str, str]: (回答, 來源法條)
    """
    # 驗證輸入
    if not question or not question.strip():
        return "請輸入問題", ""
    
    try:
        # 初始化 RAG Chain
        rag_chain_dict = initialize_rag_chain()
        
        # 執行查詢
        result = query(question, rag_chain_dict)
        
        if not result['success']:
            return result['answer'], ""
        
        # 格式化來源法條
        sources_text = ""
        if result['sources']:
            sources_list = []
            for i, source in enumerate(result['sources'], 1):
                law_name = source.get('law_name', '未知法律')
                article_no = source.get('article_no', '')
                page = source.get('page')
                score = source.get('score')
                content = source.get('content', '')

                # 限制顯示長度
                if len(content) > 300:
                    content = content[:300] + "..."

                # article_no 已是完整「第 X 條」；PDF 來源則以頁碼標示
                if article_no:
                    label = f"{law_name} {article_no}"
                elif page:
                    label = f"{law_name} 第{page}頁"
                else:
                    label = law_name
                if score:
                    label += f"（相似度 {score:.2f}）"
                sources_list.append(f"【{i}】{label}\n{content}")
            
            sources_text = "\n\n".join(sources_list)
        else:
            sources_text = "未找到相關法條"
        
        return result['answer'], sources_text
        
    except RAGError as e:
        return f"❌ 錯誤: {str(e)}", ""
    except Exception as e:
        return f"❌ 發生錯誤: {str(e)}", ""


def create_web_ui():
    """
    建立 Gradio Web UI
    
    Returns:
        gr.Blocks: Gradio UI 物件
    """
    with gr.Blocks(title="中華民國法律查詢系統") as demo:
        # 標題
        gr.Markdown("# 🏛️ 中華民國法律查詢系統")
        gr.Markdown("使用 RAG 技術提供準確的台灣法律諮詢服務")
        
        # 問題輸入
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="📝 請輸入您的問題",
                    placeholder="例如：什麼是詐欺罪？",
                    lines=3
                )
                submit_btn = gr.Button("🔍 查詢", variant="primary")
        
        # 結果顯示
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="💬 AI 回答",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                sources_output = gr.Textbox(
                    label="📚 參考法條",
                    lines=10,
                    interactive=False
                )
        
        # PDF 匯入區
        with gr.Accordion("📥 匯入 PDF 到知識庫", open=False):
            gr.Markdown(
                "上傳判決書、解釋函、法規等 PDF（按頁切分、保留頁碼）。"
                "原始檔會保存到 data/uploads/；重複匯入同一份檔案會自動覆蓋舊資料。"
            )
            upload_files = gr.File(
                label="選擇檔案",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )
            ingest_btn = gr.Button("📥 匯入知識庫", variant="secondary")
            ingest_output = gr.Textbox(
                label="匯入結果",
                lines=4,
                interactive=False
            )

        # 範例問題
        gr.Markdown(
            """
            ---
            ### 💡 範例問題
            - 什麼是詐欺罪？
            - 刑法對於竊盜的規定是什麼？
            - 民法中關於契約的規定有哪些？
            - 勞動基準法對於工時的規定？
            """
        )

        # 事件綁定
        ingest_btn.click(
            fn=upload_and_ingest,
            inputs=upload_files,
            outputs=ingest_output
        )

        submit_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        question_input.submit(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
    
    return demo


def main():
    """
    啟動 Web UI 應用程式
    """
    print("=" * 60)
    print("中華民國法律智能問答系統 - Web UI")
    print("=" * 60)
    print(f"\nOllama 模型: {config.OLLAMA_MODEL}")
    print(f"Embedding 模型: {config.EMBEDDING_MODEL}")
    print(f"Qdrant Collection: {config.QDRANT_COLLECTION}")
    
    try:
        # 建立 UI
        demo = create_web_ui()
        
        print(f"\n✓ Web UI 已啟動")
        print(f"  訪問地址: http://{config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
        print(f"  按 Ctrl+C 停止服務\n")
        
        # 啟動伺服器
        demo.launch(
            server_name=config.GRADIO_SERVER_NAME,
            server_port=config.GRADIO_SERVER_PORT,
            share=config.GRADIO_SHARE
        )
        
    except KeyboardInterrupt:
        print("\n✓ Web UI 已停止")
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        raise


if __name__ == "__main__":
    main()
