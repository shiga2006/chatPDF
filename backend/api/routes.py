import os
import json
import math
import logging
import shutil
from datetime import datetime
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.connection import get_db
from backend.models.db_models import User, Document, ChatSession, Chat
from backend.schemas.api_schemas import (
    UserRegister, UserLogin, Token, DocumentResponse, DocumentRename,
    ChatSessionResponse, ChatMessageResponse, ChatRequest, ChatResponse, CitationSchema,
    DashboardMetrics, RecentDocumentSchema, RecentQuestionSchema,
    EvaluationRequest, EvaluationResponse, EvaluationReportListItem, EvalSampleItem
)
from backend.auth.security import (
    get_current_user, hash_password, verify_password, create_access_token
)
from backend.rag.pdf_processor import extract_and_chunk_pdf
from backend.vectorstore.chroma_service import chroma_service
from backend.agents.graph import agent_graph
from backend.agents.llm_factory import get_llm
from backend.services.summary_service import generate_document_summary
from backend.models.db_models import EvaluationReport
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

router = APIRouter()

# Helper function to generate suggested questions for uploaded document
def generate_suggested_questions(filepath: str, filename: str) -> str:
    """Uses LLM to analyze the document text and generate 5 suggested questions."""
    import fitz
    try:
        doc = fitz.open(filepath)
        text_parts = []
        # Extract first 3 pages or up to 5000 characters
        for i in range(min(3, len(doc))):
            text_parts.append(doc.load_page(i).get_text())
        snippet = "\n".join(text_parts)[:5000]
        doc.close()
        
        if not snippet.strip():
            raise ValueError("Document has no text content.")
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an enterprise AI knowledge assistant. Analyze the document snippet and generate "
                "exactly 5 suggested questions that an employee might ask about this document. "
                "The questions should be specific to the document's content (e.g., policy limits, eligibility, contact info).\n"
                "Respond with ONLY a valid JSON list of strings, e.g.: [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\"]. "
                "Do NOT include any markdown block markers like ```json or any other commentary."
            )),
            ("user", "Document: {filename}\n\nSnippet:\n{snippet}\n\nSuggested Questions:")
        ])
        
        llm = get_llm(temperature=0.7)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"filename": filename, "snippet": snippet})
        
        clean_res = response.strip()
        # Clean markdown fence outputs if LLM outputted them anyway
        if clean_res.startswith("```json"):
            clean_res = clean_res.split("```json")[1].split("```")[0].strip()
        elif clean_res.startswith("```"):
            clean_res = clean_res.split("```")[1].split("```")[0].strip()
            
        questions = json.loads(clean_res)
        if isinstance(questions, list) and len(questions) > 0:
            return json.dumps(questions[:5])
    except Exception as e:
        logger.error(f"Error generating suggested questions for {filename}: {e}")
        
    # Standard fallback questions
    default_q = [
        f"What is this document ({filename}) about?",
        "What are the main rules or policies mentioned here?",
        "What are the important dates, timelines, or deadlines?",
        "Who is the point of contact mentioned in this document?",
        "Summarize the key sections of this document."
    ]
    return json.dumps(default_q)


# --- AUTHENTICATION ROUTES ---

@router.post("/register", response_model=Token)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email is already registered."
        )
        
    # Create new user
    new_user = User(
        name=user_in.name,
        email=user_in.email,
        password_hash=hash_password(user_in.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generate token
    token = create_access_token(data={"sub": new_user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "name": new_user.name,
        "email": new_user.email
    }

@router.post("/login", response_model=Token)
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if not user or not verify_password(user_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )
        
    token = create_access_token(data={"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "name": user.name,
        "email": user.email
    }


# --- DASHBOARD ROUTE ---

@router.get("/dashboard", response_model=DashboardMetrics)
def get_dashboard_metrics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Total PDFs
    total_pdfs = db.query(Document).filter(Document.user_id == current_user.id).count()
    
    # Total Conversations
    total_conversations = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).count()
    
    # Total Indexed Chunks
    total_chunks = chroma_service.count_user_chunks(current_user.id)
    
    # Recent Documents (last 5)
    recent_docs = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.upload_time.desc()).limit(5).all()
    
    recent_docs_schema = [
        RecentDocumentSchema(id=d.id, filename=d.filename, upload_time=d.upload_time)
        for d in recent_docs
    ]
    
    # Recent Questions (last 5)
    recent_chats = db.query(Chat).filter(
        Chat.user_id == current_user.id
    ).order_by(Chat.created_at.desc()).limit(5).all()
    
    recent_questions_schema = [
        RecentQuestionSchema(session_id=c.session_id, question=c.question, created_at=c.created_at)
        for c in recent_chats
    ]
    
    # Storage Used (MB)
    storage_used_bytes = 0
    user_docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    for doc in user_docs:
        if os.path.exists(doc.filepath):
            storage_used_bytes += os.path.getsize(doc.filepath)
            
    storage_used_mb = round(storage_used_bytes / (1024 * 1024), 2)
    
    return DashboardMetrics(
        total_pdfs=total_pdfs,
        total_chunks=total_chunks,
        total_conversations=total_conversations,
        recent_documents=recent_docs_schema,
        recent_questions=recent_questions_schema,
        storage_used_mb=storage_used_mb
    )


# --- DOCUMENT MANAGEMENT ROUTES ---

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Only PDF documents are allowed."
        )
        
    # Prevent duplicate file name upload for the same user
    duplicate = db.query(Document).filter(
        Document.user_id == current_user.id,
        Document.filename == file.filename
    ).first()
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A document with the name '{file.filename}' already exists."
        )
        
    # Create upload path
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_upload_dir, exist_ok=True)
    filepath = os.path.join(user_upload_dir, file.filename)
    
    # Save the file to disk
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file on disk."
        )
        
    # Create DB entry
    db_doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        filepath=filepath
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    # Run PDF processing (extraction & semantic chunking)
    try:
        chunks = extract_and_chunk_pdf(
            filepath=filepath,
            document_id=db_doc.id,
            filename=db_doc.filename,
            owner_id=current_user.id
        )
        
        if not chunks:
            # Empty document
            db.delete(db_doc)
            db.commit()
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF contains no extractable text."
            )
            
        # Index in ChromaDB
        chroma_service.add_chunks(chunks)
        
        # Automatically generate 5 suggested questions
        suggested_q = generate_suggested_questions(filepath, db_doc.filename)
        db_doc.suggested_questions = suggested_q
        db.commit()
        db.refresh(db_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed in ingestion pipeline: {e}")
        # Clean up database and disk on error
        db.delete(db_doc)
        db.commit()
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index PDF: {str(e)}"
        )
        
    return DocumentResponse(
        id=db_doc.id,
        filename=db_doc.filename,
        filepath=db_doc.filepath,
        upload_time=db_doc.upload_time,
        suggested_questions=db_doc.suggested_questions
    )

@router.get("/documents", response_model=List[DocumentResponse])
def get_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            filepath=d.filepath,
            upload_time=d.upload_time,
            suggested_questions=d.suggested_questions
        )
        for d in docs
    ]

@router.delete("/documents/{id}", status_code=status.HTTP_200_OK)
def delete_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # Delete from disk
    if os.path.exists(doc.filepath):
        try:
            os.remove(doc.filepath)
        except Exception as e:
            logger.warning(f"Could not delete physical file: {e}")
            
    # Delete from ChromaDB
    chroma_service.delete_document(doc.id)
    
    # Delete from MySQL
    db.delete(doc)
    db.commit()
    
    return {"message": f"Successfully deleted document '{doc.filename}'"}

@router.put("/documents/{id}", response_model=DocumentResponse)
def rename_document(
    id: int,
    rename_in: DocumentRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # Validate new name ends with .pdf
    new_name = rename_in.filename
    if not new_name.endswith(".pdf"):
        new_name += ".pdf"
        
    # Verify no duplication
    duplicate = db.query(Document).filter(
        Document.user_id == current_user.id,
        Document.filename == new_name,
        Document.id != id
    ).first()
    if duplicate:
        raise HTTPException(
            status_code=400,
            detail=f"A document named '{new_name}' already exists."
        )
        
    # Update on disk
    old_filepath = doc.filepath
    user_upload_dir = os.path.dirname(old_filepath)
    new_filepath = os.path.join(user_upload_dir, new_name)
    
    if os.path.exists(old_filepath):
        try:
            os.rename(old_filepath, new_filepath)
        except Exception as e:
            logger.error(f"Failed to rename physical file: {e}")
            raise HTTPException(status_code=500, detail="Failed to rename file on disk.")
    else:
        new_filepath = old_filepath  # Keep old path if missing
        
    # Update MySQL
    doc.filename = new_name
    doc.filepath = new_filepath
    db.commit()
    db.refresh(doc)
    
    # Update ChromaDB Metadata
    chroma_service.update_document_filename(doc.id, new_name)
    
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        filepath=doc.filepath,
        upload_time=doc.upload_time,
        suggested_questions=doc.suggested_questions
    )


# --- CHAT & CONVERSATION ROUTES ---

@router.post("/chat", response_model=ChatResponse)
def run_chat(
    chat_req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify or create Chat Session
    session_id = chat_req.session_id
    if not session_id:
        # Create a new session named after the first question
        truncated_title = chat_req.question[:50] + ("..." if len(chat_req.question) > 50 else "")
        session = ChatSession(
            user_id=current_user.id,
            session_name=truncated_title
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id
    else:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
            
    # Load recent chat history (last 8 messages)
    history_chats = db.query(Chat).filter(
        Chat.session_id == session_id
    ).order_by(Chat.created_at.asc()).limit(8).all()
    
    # Map history to BaseMessage lists for LangGraph
    messages = []
    for c in history_chats:
        messages.append(HumanMessage(content=c.question))
        messages.append(AIMessage(content=c.answer))
        
    # Add current question
    messages.append(HumanMessage(content=chat_req.question))
    
    # Prepare initial LangGraph state
    initial_state = {
        "messages": messages,
        "query": chat_req.question,
        "normalized_query": chat_req.question,
        "user_id": current_user.id,
        "session_id": session_id,
        "selected_document_ids": chat_req.selected_document_ids,
        "selected_domain": "general",
        "selected_tools": [],
        "subtasks": [],
        "retrieval_attempts": 0,
        "verification_score": 0.0,
        "verification_notes": "",
        "needs_reretrieval": False,
        "next_agent": "",
        "retrieved_context": [],
        "summary_result": "",
        "comparison_result": "",
        "citations": [],
        "final_answer": ""
    }
    
    try:
        # Run LangGraph Agent workflow
        output_state = agent_graph.invoke(initial_state)
        
        final_answer = output_state.get("final_answer", "")
        citations_list = output_state.get("citations", [])
        
        # Save chat exchange to DB
        new_chat = Chat(
            session_id=session_id,
            user_id=current_user.id,
            question=chat_req.question,
            answer=final_answer
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        
        # Format response
        citations_response = [
            CitationSchema(filename=c["filename"], page=c["page"], score=c["score"])
            for c in citations_list
        ]
        
        return ChatResponse(
            id=new_chat.id,
            session_id=session_id,
            session_name=session.session_name,
            question=chat_req.question,
            answer=final_answer,
            citations=citations_response,
            created_at=new_chat.created_at
        )
        
    except Exception as e:
        logger.error(f"LangGraph execution error: {e}")
        # Save a friendly error message to DB so the chat log reflects failure
        error_msg = f"Sorry, I encountered an error running the AI workflow: {str(e)}"
        new_chat = Chat(
            session_id=session_id,
            user_id=current_user.id,
            question=chat_req.question,
            answer=error_msg
        )
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        
        return ChatResponse(
            id=new_chat.id,
            session_id=session_id,
            session_name=session.session_name,
            question=chat_req.question,
            answer=error_msg,
            citations=[],
            created_at=new_chat.created_at
        )

@router.get("/history", response_model=List[dict])
def get_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns a list of all user conversations along with their messages."""
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.created_at.desc()).all()
    
    history = []
    for s in sessions:
        chats = db.query(Chat).filter(Chat.session_id == s.id).order_by(Chat.created_at.asc()).all()
        history.append({
            "id": s.id,
            "session_name": s.session_name,
            "created_at": s.created_at,
            "messages": [
                {
                    "id": c.id,
                    "question": c.question,
                    "answer": c.answer,
                    "created_at": c.created_at
                }
                for c in chats
            ]
        })
    return history


# --- EXTRA DOCUMENT UTILITIES ---

@router.get("/summary/{id}")
def get_document_summary(
    id: int,
    type: str = "detailed",
    page: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Direct route to request a document summary of type 'detailed', 'short', or 'bullet'."""
    doc = db.query(Document).filter(Document.id == id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    summary = generate_document_summary(
        document_id=doc.id,
        summary_type=type,
        page_number=page
    )
    return {"id": id, "filename": doc.filename, "summary": summary}

@router.get("/preview/{id}")
def preview_document(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Serves the raw PDF binary data for rendering previews in the frontend."""
    doc = db.query(Document).filter(Document.id == id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    if not os.path.exists(doc.filepath):
        raise HTTPException(status_code=404, detail="Physical PDF file not found on server disk.")
        
    return FileResponse(doc.filepath, media_type="application/pdf", filename=doc.filename)


# --- RAGAS EVALUATION ROUTES ---


@router.post("/evaluate", response_model=EvaluationResponse)
def run_evaluation_api(
    eval_req: EvaluationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run RAGAS evaluation on a set of question-answer-context-ground_truth samples.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY environment variable is not set. RAGAS evaluation requires an OpenAI key.",
        )

    # Convert Pydantic models to dicts
    records = []
    for s in eval_req.samples:
        records.append({
            "question": s.question,
            "answer": s.answer,
            "contexts": s.contexts,
            "ground_truth": s.ground_truth,
        })

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas import metrics as ragas_metrics

        # Resolve metrics
        metric_name_options = [
            ["faithfulness"],
            ["answer_relevancy", "answer_relevance"],
            ["context_precision"],
            ["context_recall"],
        ]
        metrics = []
        for options in metric_name_options:
            for name in options:
                m = getattr(ragas_metrics, name, None)
                if m is not None:
                    metrics.append(m)
                    break

        # Build LLM and embeddings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = LangchainLLMWrapper(ChatOpenAI(model=eval_req.judge_model, temperature=0.0))
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=eval_req.embedding_model)
        )

        dataset = Dataset.from_list(records)
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        # Parse results
        try:
            frame = result.to_pandas()
            per_sample = frame.to_dict(orient="records")
        except Exception:
            per_sample = getattr(result, "scores", [])

        if not isinstance(per_sample, list):
            per_sample = []

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "answer_relevance",
            "context_precision",
            "context_recall",
        ]
        summary = {"rows": len(per_sample)}
        for metric_name in metric_names:
            values = []
            for row in per_sample:
                value = row.get(metric_name)
                if isinstance(value, (int, float)) and not (
                    isinstance(value, float) and math.isnan(value)
                ):
                    values.append(float(value))
            if values:
                summary[metric_name] = round(sum(values) / len(values), 4)

        def sanitize(v: Any) -> Any:
            if isinstance(v, float) and math.isnan(v):
                return None
            if isinstance(v, dict):
                return {k: sanitize(v) for k, v in v.items()}
            if isinstance(v, list):
                return [sanitize(x) for x in v]
            return v

        # Save to DB
        report = EvaluationReport(
            user_id=current_user.id,
            report_name=eval_req.report_name,
            summary_json=json.dumps(summary),
            samples_json=json.dumps(sanitize(per_sample)),
            judge_model=eval_req.judge_model,
            embedding_model=eval_req.embedding_model,
            dataset_size=len(records),
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        return EvaluationResponse(
            id=report.id,
            report_name=report.report_name,
            summary=summary,
            dataset_size=report.dataset_size,
            judge_model=report.judge_model,
            embedding_model=report.embedding_model,
            created_at=report.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAGAS evaluation failed: {str(e)}",
        )


@router.get("/evaluate/reports", response_model=List[EvaluationReportListItem])
def list_evaluation_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all evaluation reports for the current user."""
    reports = (
        db.query(EvaluationReport)
        .filter(EvaluationReport.user_id == current_user.id)
        .order_by(EvaluationReport.created_at.desc())
        .all()
    )
    return [
        EvaluationReportListItem(
            id=r.id,
            report_name=r.report_name,
            dataset_size=r.dataset_size,
            created_at=r.created_at,
        )
        for r in reports
    ]


@router.get("/evaluate/reports/{report_id}")
def get_evaluation_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve a specific evaluation report with full details."""
    report = (
        db.query(EvaluationReport)
        .filter(
            EvaluationReport.id == report_id,
            EvaluationReport.user_id == current_user.id,
        )
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Evaluation report not found.")

    return {
        "id": report.id,
        "report_name": report.report_name,
        "summary": json.loads(report.summary_json),
        "samples": json.loads(report.samples_json),
        "judge_model": report.judge_model,
        "embedding_model": report.embedding_model,
        "dataset_size": report.dataset_size,
        "created_at": report.created_at.isoformat(),
    }


@router.delete("/evaluate/reports/{report_id}")
def delete_evaluation_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an evaluation report."""
    report = (
        db.query(EvaluationReport)
        .filter(
            EvaluationReport.id == report_id,
            EvaluationReport.user_id == current_user.id,
        )
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Evaluation report not found.")

    db.delete(report)
    db.commit()
    return {"message": f"Evaluation report '{report.report_name}' deleted."}
