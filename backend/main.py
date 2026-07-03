import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.connection import engine, Base
from backend.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Auto-create MySQL Tables on startup
try:
    logger.info("Creating database tables if they do not exist...")
    # Import models here to make sure they are registered on Base
    from backend.models import db_models
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified successfully.")
except Exception as e:
    logger.error(f"Failed to initialize database tables: {e}")

# Instantiate FastAPI application
app = FastAPI(
    title="Enterprise Agentic AI Knowledge Assistant API",
    description="Agentic multi-agent document search and comparison assistant powered by LangGraph, ChromaDB, and MySQL.",
    version="1.0.0"
)

# CORS configurations for cross-origin API calls (Frontend <-> Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core routes
app.include_router(api_router)

@app.get("/")
def root():
    from datetime import datetime
    return {
        "status": "online",
        "service": "Enterprise Agentic AI Knowledge Assistant API",
        "timestamp": datetime.utcnow().isoformat()
    }
