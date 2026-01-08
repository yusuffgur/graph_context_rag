import gradio as gr
import requests
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UI")

# Force IPv4 to avoid macOS localhost/IPv6 delays
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def chat(message, history):
    """
    Sends message to RAG Backend.
    """
    try:
        resp = requests.get(f"{API_URL}/query", params={"q": message}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("answer", "No answer provided.")
        else:
            return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return f"Connection Failed: {str(e)}"

def update_config(provider, api_key, model, endpoint, api_version, deployment, embedding_deployment, use_local_llm):
    """
    Updates Backend Configuration.
    """
    payload = {
        "provider": provider,
        "api_key": api_key if api_key else None,
        "model": model if model else None,
        "endpoint": endpoint if endpoint else None,
        "api_version": api_version if api_version else None,
        "deployment": deployment if deployment else None,
        "embedding_deployment": embedding_deployment if embedding_deployment else None,
        "use_local_llm": use_local_llm
    }
    
    try:
        resp = requests.post(f"{API_URL}/settings", json=payload, timeout=10)
        if resp.status_code == 200:
            return f"✅ Success! Switched to {provider}."
        else:
            return f"❌ Error {resp.status_code}: {resp.text}"
    except Exception as e:
        logger.error(f"Update Config Error: {e}")
        return f"❌ Connection Failed: {str(e)}"

def upload_files(files):
    """
    Uploads files and STREAMS status updates via SSE.
    """
    if not files:
        yield "⚠️ No files selected."
        return
    
    try:
        # 1. Upload
        yield "🚀 Uploading files..."
        uploaded_files = []
        for file_path in files:
             filename = os.path.basename(file_path)
             uploaded_files.append(("files", (filename, open(file_path, "rb"), "application/octet-stream")))
        
        
        # Increase upload timeout for large files (60s)
        resp = requests.post(f"{API_URL}/upload", files=uploaded_files, timeout=60) 
        
        for _, (_, f, _) in uploaded_files: f.close()

        if resp.status_code != 200:
            yield f"❌ Upload Error {resp.status_code}: {resp.text}"
            return

        data = resp.json()
        batch_id = data.get('batch_id')
        results = data.get('results', [])
        
        # Initial status report
        log_history = f"✅ Batch {batch_id} Created.\n"
        has_queued = False
        
        for res in results:
            fname = res.get('file')
            status = res.get('status')
            msg = res.get('message')
            
            if status == 'skipped':
                log_history += f"⏭️ SKIPPED: {fname} ({msg})\n"
            elif status == 'queued':
                log_history += f"⏳ QUEUED: {fname}\n"
                has_queued = True
        
        yield log_history

        # Only stream if we have actual work pending
        if has_queued:
            stream_url = f"{API_URL}/stream/{batch_id}"
            log_history += "🔄 Connecting to Worker Stream...\n"
            yield log_history
            
            # Stream request: Short connect timeout (5s), Infinite read timeout (None)
            with requests.get(stream_url, stream=True, timeout=(5, None)) as stream_resp:
                for line in stream_resp.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            raw_data = decoded[6:]
                            try:
                                msg = json.loads(raw_data)
                                status = msg.get("status", "")
                                progress = msg.get("progress", "")
                                log_entry = f"{status}: {progress}" if progress else status
                                log_history += log_entry + "\n"
                                yield log_history
                            except:
                                pass
        else:
            log_history += "✅ All files handled.\n"
            yield log_history

    except Exception as e:
        logger.error(f"Upload Error: {e}")
        yield f"❌ Process Failed: {str(e)}"

def load_config():
    """
    Fetches current backend configuration to populate UI.
    """
    try:
        logger.info(f"Loading config from {API_URL}/settings...")
        # Short timeout to fail fast if backend isn't ready
        resp = requests.get(f"{API_URL}/settings", timeout=2)
        
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Config loaded: {data}")
            return (
                data.get("provider", "openai"),
                data.get("api_key", ""),
                data.get("model", ""),
                data.get("endpoint", ""),
                data.get("api_version", ""),
                data.get("deployment", ""),
                data.get("embedding_deployment", ""),
                data.get("use_local_llm", False),
                "✅ Configuration Loaded from Backend"
            )
        else:
             return ("openai", "", "", "", "", "", "", False, f"❌ Failed to load config: {resp.text}")
    except Exception as e:
        logger.error(f"Config Load Error: {e}")
        return ("openai", "", "", "", "", "", "", False, f"⚠️ Backend not reachable: {e}")

# -- UI Layout --
with gr.Blocks(title="Federated RAG Console") as demo:
    gr.Markdown("# 🧠 Federated RAG Console")
    
    with gr.Tabs():
        # TAB 1: Chat
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=600, label="Conversation")
                    msg = gr.Textbox(show_label=False, placeholder="Ask a question about your documents...", container=False)
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Chat Context")
                    with gr.Row():
                        context_dropdown = gr.Dropdown(
                            choices=["All Documents"], 
                            value="All Documents", 
                            label="Scope", 
                            interactive=True,
                            scale=3
                        )
                        refresh_docs_btn = gr.Button("🔄", scale=1)
                    
                    mode_radio = gr.Radio(
                        choices=["Hybrid (Default)", "Context Only", "Graph Only"],
                        value="Hybrid (Default)",
                        label="Retrieval Mode",
                        interactive=True
                    )
                    
                    gr.Markdown("### 📚 Context & PDFs")
                    citations_html = gr.HTML(label="Sources")

            def refresh_documents():
                """Fetch available documents from API."""
                try:
                    resp = requests.get(f"{API_URL}/documents", timeout=5)
                    if resp.status_code == 200:
                        docs = resp.json().get("documents", [])
                        # Format: "Name" but value is "ID"
                        # Gradio Dropdown choices can be list of tuples (name, value) or simple list
                        # Let's use simple list of names for now, but we need ID for filtering.
                        # Actually Gradio support (name, value) tuples.
                        choices = [("All Documents", None)] + [(d['name'], d['id']) for d in docs]
                        return gr.update(choices=choices, value=None)
                except Exception as e:
                    logger.error(f"Doc Refresh Error: {e}")
                return gr.update()

            def sanitize_history(history):
                if not history: return []
                sanitized = []
                for item in history:
                    if isinstance(item, (list, tuple)):
                        if len(item) >= 1 and item[0]: sanitized.append({"role": "user", "content": item[0]})
                        if len(item) >= 2 and item[1]: sanitized.append({"role": "assistant", "content": item[1]})
                    elif isinstance(item, dict):
                        sanitized.append(item)
                return sanitized

            # State for safe data passing
            user_question_state = gr.State("")

            def user_msg(user_message, history):
                # Update history and store message in state
                new_hist = sanitize_history(history) + [{"role": "user", "content": user_message}]
                return "", new_hist, user_message

            def bot_msg(history, doc_filter, mode_label, current_q):
                """Chat with optional document filter."""
                if not current_q:
                    # Fallback to history if state is empty (shouldn't happen)
                    if not history: return history, ""
                    current_q = history[-1]['content']

                # Map UI Label to API Value
                mode_map = {
                    "Hybrid (Default)": "hybrid",
                    "Context Only": "vector",
                    "Graph Only": "graph"
                }
                api_mode = mode_map.get(mode_label, "hybrid")

                # Append placeholder
                history.append({"role": "assistant", "content": "Thinking..."})
                yield history, ""
                
                try:
                    params = {"q": current_q, "mode": api_mode}
                    # dropdown value is None (All) or "temp/filename"
                    if doc_filter and doc_filter != "All Documents":
                        params["filter"] = doc_filter

                    resp = requests.get(f"{API_URL}/query", params=params, timeout=300)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "No answer provided.")
                        sources = data.get("sources", [])
                        
                        # Update the last message (assistant's response)
                        history[-1]['content'] = answer
                        
                        # Build Rich HTML for Citations
                        html_content = "<div style='font-size: 0.9em;'>"
                        
                        # DEBUG PANEL
                        debug_data = data.get("debug", {})
                        if debug_data:
                            html_content += f"""
                            <details style="margin-bottom: 15px; border: 1px solid #eee; padding: 5px; border-radius: 5px;">
                                <summary style="cursor: pointer; color: #777;">🔍 Debug Retrieval (Click to Expand)</summary>
                                <pre style="font-size: 0.7em; background: #333; color: #0f0; padding: 10px; overflow-x: auto;">
Provider: {debug_data.get('llm_provider', 'Unknown')}
Original: "{debug_data.get('original_query', '')}"
Refined:  "{debug_data.get('refined_query', '')}"
Candidates: {debug_data.get('vector_candidates')} Vector -> {debug_data.get('reranked_candidates')} Reranked
                                
FINAL PROMPT SENT TO LLM:
-------------------------
{debug_data.get('final_prompt')}
                                </pre>
                            </details>
                            """

                        for i, src in enumerate(sources):
                            file_path = src.get('source', 'Unknown')
                            filename = os.path.basename(file_path)
                            
                            # Construct URL: Remove 'temp/' prefix if present due to mounting logic
                            safe_name = os.path.basename(file_path) 
                            # If batch ID is prefixed (UUID_Name.pdf), it works as is.
                            page = src.get('page_number', 1)
                            score = src.get('score', 0)
                            text = src.get('text', '')[:250].replace("<", "&lt;").replace(">", "&gt;")
                            
                            # Highlight: Use Chrome's #page=N and #search="term"
                            # search="term" highlights the first occurrence
                            # Extract first 3-4 significant words for highlighting to avoid URL length issues
                            search_term = " ".join(text.split()[:5]).replace('"', '')
                            
                            pdf_url = f"{API_URL}/files/{safe_name}#page={page}"
                            
                            # Using iframe for in-screen view
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 15px; border-radius: 8px; background-color: #f9f9f9;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                    <strong>📄 {filename}</strong> 
                                    <span style="background-color: #e0f0ff; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;">Page {page} • Score: {score:.2f}</span>
                                </div>
                                <p style="font-size: 0.85em; color: #555; font-style: italic; border-left: 3px solid #007bff; padding-left: 8px; margin: 8px 0;">"{text}..."</p>
                                
                                <!-- PDF Embed -->
                                <iframe src="{pdf_url}" width="100%" height="350px" style="border: 1px solid #ccc; border-radius: 4px;">
                                    <p>Your browser does not support iframes. <a href="{pdf_url}" target="_blank">Download PDF</a></p>
                                </iframe>
                                <div style="text-align: right; margin-top: 5px;">
                                     <a href="{pdf_url}" target="_blank" style="font-size: 0.8em; text-decoration: none; color: #007bff;">↗️ Open Full Screen</a>
                                </div>
                            </div>
                            """
                        html_content += "</div>"
                        
                        yield history, html_content
                    else:
                        error_msg = f"Error {resp.status_code}: {resp.text}"
                        history[-1]['content'] = error_msg
                        yield history, f"<div style='color: red;'>Backend Error: {error_msg}</div>"
                except Exception as e:
                    logger.error(f"Chat Error: {e}")
                    error_msg = f"Connection Failed: {str(e)}"
                    history[-1]['content'] = error_msg
                    yield history, f"<div style='color: red;'>Connection Error: {error_msg}</div>"

            # Wire up events
            msg.submit(user_msg, [msg, chatbot], [msg, chatbot, user_question_state], queue=False).then(
                bot_msg, [chatbot, context_dropdown, mode_radio, user_question_state], [chatbot, citations_html]
            )
            send_btn.click(user_msg, [msg, chatbot], [msg, chatbot, user_question_state], queue=False).then(
                bot_msg, [chatbot, context_dropdown, mode_radio, user_question_state], [chatbot, citations_html]
            )
            clear_btn.click(lambda: None, None, chatbot, queue=False)
            clear_btn.click(lambda: "", None, citations_html, queue=False)
            
            # Context Refresh
            refresh_docs_btn.click(refresh_documents, outputs=[context_dropdown])
            demo.load(refresh_documents, outputs=[context_dropdown])

        # TAB 2: Ingestion
        with gr.TabItem("📄 Ingestion"):
            gr.Markdown("### 📤 Upload Documents")
            gr.Markdown("Supported formats: PDF, TXT. Files are processed asynchronously.")
            
            file_input = gr.File(file_count="multiple", label="Select Files")
            upload_btn = gr.Button("🚀 Upload & Process", variant="primary")
            upload_status = gr.Textbox(label="Real-time Status", interactive=False, lines=20)
            
            gr.Markdown("---")
            gr.Markdown("### 🧹 System Management")
            reset_btn = gr.Button("⚠️ Reset Knowledge Base (Clear All Data)", variant="stop")
            reset_status = gr.Textbox(label="Reset Status", interactive=False)

            def reset_system():
                try:
                    resp = requests.post(f"{API_URL}/reset", timeout=10)
                    if resp.status_code == 200:
                        return "✅ System Reset Successful. All data cleared."
                    else:
                        return f"❌ Reset Failed: {resp.text}"
                except Exception as e:
                    return f"❌ Connection Error: {str(e)}"

            # Wire up Upload with Refresh
            upload_btn.click(
                fn=upload_files,
                inputs=[file_input],
                outputs=[upload_status]
            ).then(
                fn=refresh_documents,
                outputs=[context_dropdown]
            )
            
            reset_btn.click(reset_system, None, [reset_status])

        # TAB 3: Settings
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### 🔧 LLM Provider Configuration")
            gr.Markdown("Dynamically switch providers without restarting the server.")
            
            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    choices=["openai", "azure", "gemini", "ollama"], 
                    value="openai", 
                    label="Provider"
                )
                model_input = gr.Textbox(label="Model Name (Optional)", placeholder="e.g. gpt-4o, mistral")
            
            api_key_input = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
            
            with gr.Group():
                gr.Markdown("#### Azure OpenAI Specifics")
                endpoint_input = gr.Textbox(label="Azure Endpoint", placeholder="https://your-resource.openai.azure.com/")
                api_version_input = gr.Textbox(label="API Version", placeholder="2023-12-01-preview")
                deployment_input = gr.Textbox(label="Chat Deployment Name", placeholder="gpt-35-turbo")
                embedding_input = gr.Textbox(label="Embedding Deployment Name", placeholder="text-embedding-ada-002")

            with gr.Row():
                 use_local_llm_input = gr.Checkbox(label="Use Local LLM (Ollama) for Background Tasks", value=False, info="If unchecked, the worker will SKIP the local model and use the Cloud Provider for Graph/Vector tasks.")

            # Add a Refresh Button to manually pull config if needed
            with gr.Row():
                update_btn = gr.Button("Update Configuration", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh Configuration", variant="secondary")
            
            status_output = gr.Textbox(label="Status", interactive=False, lines=5)

            update_btn.click(
                fn=update_config,
                inputs=[
                    provider_dropdown, api_key_input, model_input, 
                    endpoint_input, api_version_input, deployment_input, embedding_input,
                    use_local_llm_input
                ],
                outputs=[status_output]
            )
            
            # Hook refresh button
            refresh_btn.click(
                fn=load_config,
                outputs=[
                    provider_dropdown, api_key_input, model_input, 
                    endpoint_input, api_version_input, deployment_input, embedding_input,
                    use_local_llm_input,
                    status_output
                ]
            )

            # Load on Launch
            demo.load(
                fn=load_config,
                outputs=[
                    provider_dropdown, api_key_input, model_input, 
                    endpoint_input, api_version_input, deployment_input, embedding_input,
                    use_local_llm_input,
                    status_output
                ]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
