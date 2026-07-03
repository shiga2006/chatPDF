import streamlit as st
import json
from frontend.utils import (
    get_chat_history_api, send_chat_message_api, get_uploaded_documents
)

def show_chat():
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 20px;'>Enterprise Knowledge Assistant</h2>", unsafe_allow_html=True)
    
    # 1. Fetch history and documents list
    history = get_chat_history_api()
    docs = get_uploaded_documents()
    
    # Initialize session state variables for chat
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "chat_input_val" not in st.session_state:
        st.session_state.chat_input_val = ""
        
    # --- RENDER SIDEBAR CONTROLS ---
    st.sidebar.markdown("<h4 style='color: #a78bfa;'>Conversations</h4>", unsafe_allow_html=True)
    
    # New Chat Button
    if st.sidebar.button("➕ New Chat Session", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.selected_docs_scope = []
        st.toast("Started a new conversation session.")
        st.rerun()
        
    # List active chat sessions
    if not history:
        st.sidebar.info("No active chat sessions.")
    else:
        # Create dictionary for select list
        sessions_map = {s["id"]: s["session_name"] for s in history}
        
        # Determine index of current session
        current_id = st.session_state.current_session_id
        session_ids = list(sessions_map.keys())
        
        selected_sid = st.sidebar.radio(
            "Select a session",
            options=session_ids,
            format_func=lambda x: f"💬 {sessions_map[x][:26]}...",
            index=session_ids.index(current_id) if current_id in session_ids else 0,
            label_visibility="collapsed"
        )
        
        if selected_sid != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_sid
            st.rerun()
            
    st.sidebar.markdown("<hr style='border-color: #33334d; margin: 15px 0;'/>", unsafe_allow_html=True)
    st.sidebar.markdown("<h4 style='color: #60a5fa;'>Query Scope (PDF Filter)</h4>", unsafe_allow_html=True)
    st.sidebar.caption("Filter retrieval to specific files. If none selected, the assistant queries all your uploaded documents.")
    
    # Multiselect for documents scope
    selected_doc_ids = []
    if docs:
        selected_doc_names = st.sidebar.multiselect(
            "Target Documents",
            options=[d["filename"] for d in docs],
            default=[],
            label_visibility="collapsed"
        )
        # Map back to IDs
        selected_doc_ids = [d["id"] for d in docs if d["filename"] in selected_doc_names]
    else:
        st.sidebar.caption("⚠️ No documents uploaded yet. Upload a PDF on the Documents page.")

    # --- MAIN VIEW WINDOW ---
    
    # Find current session messages
    current_session = None
    if st.session_state.current_session_id and history:
        current_session = next((s for s in history if s["id"] == st.session_state.current_session_id), None)
        
    messages = []
    if current_session:
        messages = current_session.get("messages", [])
        st.caption(f"Session ID: #{current_session['id']} &mdash; Created: {current_session['created_at'][:10]}")
    else:
        st.caption("New Conversation Session &mdash; Temporary memory active")
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Render messages
    for msg in messages:
        # User message
        with st.chat_message("user"):
            st.write(msg["question"])
            
        # Assistant message
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            
            # Since stored DB history doesn't cache citations (in the basic table),
            # we can show a placeholder or note, or if we want, we can save/retrieve citations from DB.
            # In our db schema, Chats stores question/answer. For history chats, we can let user know
            # that citations are detailed during live streams.
            st.markdown("<div style='font-size: 11px; color: #555; text-align: right;'>Archived Message</div>", unsafe_allow_html=True)

    # If user selected a document, show suggested questions
    suggested_questions = []
    if selected_doc_ids and docs:
        # Load suggested questions of the first selected document
        first_selected_doc = next((d for d in docs if d["id"] == selected_doc_ids[0]), None)
        if first_selected_doc and first_selected_doc.get("suggested_questions"):
            try:
                suggested_questions = json.loads(first_selected_doc["suggested_questions"])
            except Exception:
                pass
    elif docs and len(docs) > 0:
        # Default questions from first document overall
        try:
            suggested_questions = json.loads(docs[0]["suggested_questions"])
        except Exception:
            pass
            
    # Render Suggested Questions
    if suggested_questions and len(messages) == 0:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='color: #a0a0c0; font-size: 13px; font-weight: 600;'>💡 Suggested Questions:</span>", unsafe_allow_html=True)
        
        # Grid layout for suggestion buttons
        q_cols = st.columns(2)
        for i, q in enumerate(suggested_questions[:4]):  # Show up to 4 questions
            col = q_cols[i % 2]
            if col.button(q, key=f"sug_btn_{i}", use_container_width=True, type="secondary"):
                st.session_state.chat_input_val = q
                st.rerun()

    # --- CHAT INPUT & EXECUTION ---
    
    # Prompt Input
    prompt = st.chat_input("Ask a question about the uploaded document policies...")
    
    # Or if a suggestion was clicked
    if st.session_state.chat_input_val:
        prompt = st.session_state.chat_input_val
        st.session_state.chat_input_val = "" # Reset
        
    if prompt:
        # Display user message instantly
        with st.chat_message("user"):
            st.write(prompt)
            
        # Display typing indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            citations_placeholder = st.empty()
            message_placeholder.markdown("*Thinking... Analyzing documents...*")
            
            # Call backend API
            with st.spinner(""):
                response = send_chat_message_api(
                    question=prompt,
                    session_id=st.session_state.current_session_id,
                    selected_document_ids=selected_doc_ids
                )
                
            if response:
                # Update text response
                message_placeholder.markdown(response["answer"])
                
                # Render Citations expander
                cits = response.get("citations", [])
                if cits:
                    with citations_placeholder.container():
                        with st.expander("📚 View Citations & Reference Sources", expanded=False):
                            for c in cits:
                                score_pct = int(c["score"] * 100)
                                st.markdown(
                                    f"- **{c['filename']}** (Page {c['page']}) &mdash; "
                                    f"*Confidence match: `{score_pct}%`*"
                                )
                
                # If this was a new session, update state and rerun to load sidebar session ID
                if st.session_state.current_session_id is None:
                    st.session_state.current_session_id = response["session_id"]
                    st.rerun()
            else:
                message_placeholder.markdown("⚠️ Sorry, I could not generate a response. Please check if backend services are running.")
                
    # --- FOOTER BUTTONS ---
    if messages:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                # We can call delete session or similar. We will just start a new session locally.
                st.session_state.current_session_id = None
                st.toast("Cleared active session. Previous sessions are saved in history.")
                st.rerun()
        with btn_col2:
            st.markdown("<div style='text-align: right;'><a href='/#/Documents' target='_self'><button style='background-color:#1e1e2f; color:#a78bfa; border:1px solid #33334d; padding:8px 16px; border-radius:6px; cursor:pointer; width:100%;'>📁 Upload Document</button></a></div>", unsafe_allow_html=True)
