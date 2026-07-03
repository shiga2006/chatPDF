import os
import requests
import streamlit as st
from typing import Dict, Any, List, Optional

# Backend URL config - adjust if needed
API_URL = os.getenv("API_URL", "http://localhost:8000")

def api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    auth_required: bool = True
) -> requests.Response:
    """
    Standardized API client method with authorization header injections.
    Clears local session if 401 Unauthorized is returned.
    """
    url = f"{API_URL}{endpoint}"
    headers = {}
    
    if auth_required:
        if "token" in st.session_state and st.session_state.token:
            headers["Authorization"] = f"Bearer {st.session_state.token}"
        else:
            # Token missing, clear auth state
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()
            
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            json=json_data,
            files=files,
            timeout=300  # High timeout for RAG / LLM generation
        )
        
        if response.status_code == 401:
            # Session expired
            st.session_state.token = None
            st.session_state.user = None
            st.error("Session expired. Please log in again.")
            st.rerun()
            
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection error: {e}")
        # Create a mock failed response
        mock_response = requests.Response()
        mock_response.status_code = 500
        return mock_response

def register_user(name: str, email: str, password: str) -> Optional[Dict[str, Any]]:
    """Register a new user account."""
    response = api_request(
        "POST",
        "/register",
        json_data={"name": name, "email": email, "password": password},
        auth_required=False
    )
    if response.status_code == 200:
        return response.json()
    else:
        try:
            detail = response.json().get("detail", "Registration failed.")
        except ValueError:
            detail = "Registration failed."
        st.error(detail)
        return None

def login_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user."""
    response = api_request(
        "POST",
        "/login",
        json_data={"email": email, "password": password},
        auth_required=False
    )
    if response.status_code == 200:
        return response.json()
    else:
        try:
            detail = response.json().get("detail", "Login failed. Check your email or password.")
        except ValueError:
            detail = "Login failed."
        st.error(detail)
        return None

def get_dashboard_metrics() -> Optional[Dict[str, Any]]:
    """Fetch aggregated statistics for the current user."""
    response = api_request("GET", "/dashboard")
    if response.status_code == 200:
        return response.json()
    return None

def get_uploaded_documents() -> List[Dict[str, Any]]:
    """Fetch the list of user documents."""
    response = api_request("GET", "/documents")
    if response.status_code == 200:
        return response.json()
    return []

def upload_document_api(file_name: str, file_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Upload and process a PDF file."""
    files = {"file": (file_name, file_bytes, "application/pdf")}
    response = api_request("POST", "/upload", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        try:
            detail = response.json().get("detail", "Failed to upload document.")
        except ValueError:
            detail = f"Failed to upload document ({response.status_code})"
        st.error(detail)
        return None

def delete_document_api(doc_id: int) -> bool:
    """Delete a document by ID."""
    response = api_request("DELETE", f"/documents/{doc_id}")
    if response.status_code == 200:
        return True
    return False

def rename_document_api(doc_id: int, new_filename: str) -> Optional[Dict[str, Any]]:
    """Rename a document by ID."""
    response = api_request("PUT", f"/documents/{doc_id}", json_data={"filename": new_filename})
    if response.status_code == 200:
        return response.json()
    return None

def send_chat_message_api(
    question: str,
    session_id: Optional[int] = None,
    selected_document_ids: Optional[List[int]] = None
) -> Optional[Dict[str, Any]]:
    """Sends user chat message to LangGraph assistant."""
    response = api_request(
        "POST",
        "/chat",
        json_data={
            "question": question,
            "session_id": session_id,
            "selected_document_ids": selected_document_ids
        }
    )
    if response.status_code == 200:
        return response.json()
    return None

def get_chat_history_api() -> List[Dict[str, Any]]:
    """Retrieve chat logs and session history."""
    response = api_request("GET", "/history")
    if response.status_code == 200:
        return response.json()
    return []

def get_document_summary_api(doc_id: int, summary_type: str = "detailed") -> str:
    """Retrieve summary of document."""
    response = api_request("GET", f"/summary/{doc_id}?type={summary_type}")
    if response.status_code == 200:
        return response.json().get("summary", "Summary not generated.")
    return "Error fetching summary."

def get_pdf_preview_bytes(doc_id: int) -> Optional[bytes]:
    """Fetch raw PDF bytes from server to render in preview pane."""
    response = api_request("GET", f"/preview/{doc_id}")
    if response.status_code == 200:
        return response.content
    return None
