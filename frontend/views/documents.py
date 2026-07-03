import json
import base64
import streamlit as st
from datetime import datetime
from frontend.utils import (
    get_uploaded_documents, upload_document_api, delete_document_api,
    rename_document_api, get_document_summary_api, get_pdf_preview_bytes
)

def show_documents():
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 20px;'>Document Workspace Manager</h2>", unsafe_allow_html=True)
    
    # Refresh documents list
    docs = get_uploaded_documents()
    
    # Split layout: Left column lists and allows uploads, Right column manages selected document
    left_panel, right_panel = st.columns([5, 7])
    
    selected_doc_id = None
    
    with left_panel:
        st.markdown("<h4 style='color: #a78bfa; margin-bottom: 12px;'>Upload New PDF</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a document (PDF only, max 50MB)",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Upload document button
            if st.button("Process & Index Document", use_container_width=True, type="primary"):
                with st.spinner("Extracting text, building semantic chunks, generating embeddings, and storing in ChromaDB..."):
                    res = upload_document_api(uploaded_file.name, uploaded_file.getvalue())
                    if res:
                        st.success(f"Successfully processed: '{uploaded_file.name}'!")
                        st.balloons()
                        # Refresh page to show new document
                        st.rerun()
                        
        st.markdown("<hr style='border-color: #33334d; margin: 25px 0;'/>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 12px;'>Uploaded Documents</h4>", unsafe_allow_html=True)
        
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            # Custom styled list selector
            doc_options = {d["id"]: d["filename"] for d in docs}
            
            # Use streamlit selectbox or radio for selection
            # We'll use a neat radio selection styled with css if possible, but selectbox is most reliable.
            selected_doc_id = st.selectbox(
                "Select a document to manage",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                label_visibility="collapsed",
                key="doc_selector"
            )
            
            # Print list details
            for doc in docs:
                is_selected = doc["id"] == selected_doc_id
                bg_color = "#232338" if is_selected else "#141424"
                border_color = "#8b5cf6" if is_selected else "#33334d"
                
                try:
                    dt = datetime.fromisoformat(doc["upload_time"].replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%b %d, %Y")
                except Exception:
                    formatted_time = doc["upload_time"]
                    
                st.markdown(f"""
                <div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 13px;">{doc['filename']}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                        <span style="color: #7c7c9c; font-size: 11px;">ID: #{doc['id']}</span>
                        <span style="color: #7c7c9c; font-size: 11px;">{formatted_time}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with right_panel:
        if selected_doc_id is None:
            st.markdown(
                "<div style='height: 100%; display: flex; align-items: center; justify-content: center; border: 2px dashed #33334d; border-radius: 12px; padding: 40px; text-align: center; color: #7c7c9c;'>"
                "Select an indexed document from the left list to view summaries, suggested questions, and manage actions."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            # Get selected document metadata
            doc = next(d for d in docs if d["id"] == selected_doc_id)
            
            st.markdown(f"<h3 style='margin: 0; font-weight: 700;'>{doc['filename']}</h3>", unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # --- Quick Actions (Rename & Delete) ---
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                with st.popover("✏️ Rename Document", use_container_width=True):
                    new_name = st.text_input("New Filename", value=doc["filename"])
                    if st.button("Apply Rename", use_container_width=True):
                        if new_name.strip() and new_name != doc["filename"]:
                            res = rename_document_api(doc["id"], new_name)
                            if res:
                                st.success("Document renamed!")
                                st.rerun()
                                
            with action_col2:
                if st.button("🗑️ Delete Document", type="secondary", use_container_width=True):
                    if delete_document_api(doc["id"]):
                        st.toast("Document deleted successfully!")
                        st.rerun()
                        
            st.markdown("<hr style='border-color: #33334d; margin: 15px 0;'/>", unsafe_allow_html=True)
            
            # --- Suggested Questions ---
            st.markdown("<h5 style='color: #60a5fa;'>Auto-Generated Suggested Questions</h5>", unsafe_allow_html=True)
            try:
                suggested_json = doc.get("suggested_questions")
                if suggested_json:
                    questions = json.loads(suggested_json)
                    for q in questions:
                        st.markdown(f"<div style='background-color: #141424; padding: 8px 12px; border-radius: 6px; border-left: 2px solid #3b82f6; margin-bottom: 6px; font-size: 13px;'>{q}</div>", unsafe_allow_html=True)
                else:
                    st.info("No suggested questions cached.")
            except Exception as e:
                st.error("Error displaying suggested questions.")
                
            st.markdown("<hr style='border-color: #33334d; margin: 15px 0;'/>", unsafe_allow_html=True)
            
            # --- Summary Generator ---
            st.markdown("<h5 style='color: #a78bfa;'>Document Summarization Engine</h5>", unsafe_allow_html=True)
            
            summary_col1, summary_col2 = st.columns([2, 1])
            with summary_col1:
                sum_type = st.selectbox(
                    "Select Summary Type",
                    options=["detailed", "short", "bullet"],
                    format_func=lambda x: x.capitalize(),
                    label_visibility="collapsed"
                )
            with summary_col2:
                generate_sum = st.button("Generate", use_container_width=True, type="primary")
                
            if generate_sum:
                with st.spinner(f"Compiling {sum_type} summary using LLM..."):
                    summary_text = get_document_summary_api(doc["id"], sum_type)
                    st.markdown(f"""
                    <div style="background-color: #1a1a2e; border: 1px solid #4a347d; border-radius: 8px; padding: 16px; margin-top: 15px; font-size: 13.5px; max-height: 350px; overflow-y: auto;">
                        {summary_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<hr style='border-color: #33334d; margin: 15px 0;'/>", unsafe_allow_html=True)
            
            # --- Interactive PDF Previewer ---
            with st.expander("👁️ Toggle Interactive PDF Preview", expanded=False):
                with st.spinner("Downloading PDF file from backend server..."):
                    pdf_bytes = get_pdf_preview_bytes(doc["id"])
                    
                if not pdf_bytes:
                    st.error("Could not preview PDF. Check backend storage.")
                else:
                    try:
                        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="550" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error rendering PDF: {e}")
