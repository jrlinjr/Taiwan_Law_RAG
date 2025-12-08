"""
Gradio Web UI 應用程式
提供友善的網頁介面進行法律問答

主要功能：
- initialize_rag_chain(): 初始化 RAG 系統
- answer_question(): 處理使用者問題
- create_web_ui(): 建立 Gradio UI
- main(): 啟動應用程式
"""

import gradio as gr
from typing import Tuple

from rag import create_rag_chain, query, RAGError
from config import config


# 全域 RAG Chain 快取
_rag_chain_dict = None


def initialize_rag_chain():
    """
    初始化 RAG 系統
    
    使用全域快取避免重複初始化，提升效能。
    
    Returns:
        Dict: 包含 chain 和 retriever 的字典
    """
    global _rag_chain_dict
    
    if _rag_chain_dict is None:
        _rag_chain_dict = create_rag_chain()
    
    return _rag_chain_dict


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
                content = source.get('content', '')
                
                # 限制顯示長度
                if len(content) > 300:
                    content = content[:300] + "..."
                
                label = f"{law_name} 第{article_no}條" if article_no else law_name
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
