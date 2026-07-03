import requests
import streamlit as st
from frontend.utils import API_URL

def show_settings():
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 20px;'>System Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #a78bfa; margin-bottom: 12px;'>AI Engine Status</h4>", unsafe_allow_html=True)
    
    # Check Backend Connection
    try:
        res = requests.get(f"{API_URL}/", timeout=3)
        backend_alive = res.status_code == 200
    except Exception:
        backend_alive = False
        
    status_color = "#34d399" if backend_alive else "#f87171"
    status_text = "Connected" if backend_alive else "Disconnected (Check if backend server is running on port 8000)"
    
    st.markdown(f"""
    <div style="background-color: #1a1a2e; padding: 15px 20px; border-radius: 8px; border: 1px solid #33334d; margin-bottom: 20px;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #232338; height: 35px;">
                <td style="font-weight: 600; width: 40%;">FastAPI Backend URL</td>
                <td style="color: #7c7c9c; font-family: monospace;">{API_URL}</td>
            </tr>
            <tr style="border-bottom: 1px solid #232338; height: 35px;">
                <td style="font-weight: 600;">Backend Status</td>
                <td style="font-weight: bold; color: {status_color};">{status_text}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Local LLM (Ollama) Health Check
    st.markdown("<h5 style='color: #60a5fa;'>Local Ollama Engine Connection</h5>", unsafe_allow_html=True)
    st.caption("Verifies if the Ollama service is reachable on your system.")
    
    ollama_check = st.button("Test Ollama Connection", use_container_width=True)
    if ollama_check:
        try:
            # Check default Ollama URL
            ollama_url = "http://localhost:11434"
            res = requests.get(ollama_url, timeout=3)
            if res.status_code == 200:
                st.success("Ollama is running successfully on http://localhost:11434!")
                # Get loaded models
                models_res = requests.get(f"{ollama_url}/api/tags", timeout=3)
                if models_res.status_code == 200:
                    models = [m["name"] for m in models_res.json().get("models", [])]
                    if models:
                        st.info(f"Available Local Models: {', '.join(models)}")
                    else:
                        st.warning("Ollama is running but no models were found. Run 'ollama pull llama3.2' in terminal.")
            else:
                st.error("Ollama is reachable but returned an unexpected status code.")
        except Exception as e:
            st.error(f"Cannot reach Ollama on http://localhost:11434. Make sure you started the Ollama app. Error: {e}")
            
    st.markdown("<hr style='border-color: #33334d; margin: 30px 0;'/>", unsafe_allow_html=True)
    
    # System Architecture Diagram / Educational panel
    st.markdown("<h4 style='color: #a78bfa;'>How the Agentic Architecture Works</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    This application utilizes **LangGraph** to build a Stateful Multi-Agent system that routes your inquiries to dedicated specialist units.
    
    1. **Memory Node**: Receives your query, looks up MySQL history, and if you are asking a context-dependent question (e.g. *"explain it simply"* after asking *"what is the leave policy?"*), it automatically rewrites the query into a standalone sentence containing the reference.
    
    2. **Supervisor Agent**: Reads the (rewritten) query and decides the execution path.
       - **Retrieval Agent**: Queried if you ask a fact-finding question. Performs semantic vector query on ChromaDB to pull matching chunk details.
       - **Summary Agent**: Activated if you request a summary of page or document contents. Fetches chunks sequentially to build structured summaries.
       - **Comparison Agent**: Triggered when you ask to compare multiple documents. Pulls data from both, then compiles side-by-side tabular lists.
       
    3. **Citation Agent**: The final step in the pipeline. It reads the generated text and context details to compile precise citations mapping to the original PDF filename, page number, and vector match confidence score.
    """)
