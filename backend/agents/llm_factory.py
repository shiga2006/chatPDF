import logging
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from backend.config import settings

logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Returns the configured LLM instance (Ollama or OpenAI).
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set but LLM_PROVIDER is 'openai'. Falling back to local Ollama.")
        else:
            logger.info(f"Instantiating OpenAI ChatModel with model: {settings.LLM_MODEL or 'gpt-4o-mini'}")
            return ChatOpenAI(
                model=settings.LLM_MODEL or "gpt-4o-mini",
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
    # Default is Ollama
    logger.info(f"Instantiating Ollama ChatModel at {settings.OLLAMA_HOST} with model: {settings.LLM_MODEL}")
    return ChatOllama(
        base_url=settings.OLLAMA_HOST,
        model=settings.LLM_MODEL,
        temperature=temperature
    )
