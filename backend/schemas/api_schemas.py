from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

# Auth Schemas
class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str
    email: str

# Document Schemas
class DocumentResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    upload_time: datetime
    suggested_questions: Optional[str] = None

    class Config:
        from_attributes = True

class DocumentRename(BaseModel):
    filename: str

# Chat Schemas
class ChatSessionResponse(BaseModel):
    id: int
    session_name: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    question: str
    # If provided, only search in these documents
    selected_document_ids: Optional[List[int]] = None

class CitationSchema(BaseModel):
    filename: str
    page: int
    score: float

class ChatResponse(BaseModel):
    id: int  # Message ID
    session_id: int
    session_name: str
    question: str
    answer: str
    citations: List[CitationSchema]
    created_at: datetime

# Dashboard Schemas
class RecentDocumentSchema(BaseModel):
    id: int
    filename: str
    upload_time: datetime

class RecentQuestionSchema(BaseModel):
    session_id: int
    question: str
    created_at: datetime

class DashboardMetrics(BaseModel):
    total_pdfs: int
    total_chunks: int
    total_conversations: int
    recent_documents: List[RecentDocumentSchema]
    recent_questions: List[RecentQuestionSchema]
    storage_used_mb: float
