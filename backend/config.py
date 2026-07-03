import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+pymysql://root:rootpassword@localhost:3306/chatpdf_db"
    JWT_SECRET: str = "supersecretjwtkeyforagenticassistant123!@#"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    LLM_PROVIDER: str = "ollama"
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    OPENAI_API_KEY: str = ""
    CHROMA_DIR: str = "chromadb"
    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.CHROMA_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
