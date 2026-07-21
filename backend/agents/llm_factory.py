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
    model_name = settings.LLM_MODEL or "llama3.2"
    logger.info(f"Instantiating Ollama ChatModel at {settings.OLLAMA_HOST} with model: {model_name}")

    try:
        return ChatOllama(
            base_url=settings.OLLAMA_HOST,
            model=model_name,
            temperature=temperature
        )
    except Exception as exc:
        if settings.OPENAI_API_KEY:
            logger.warning(f"Ollama model '{model_name}' is unavailable ({exc}); falling back to OpenAI.")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY
            )

        raise RuntimeError(
            f"LLM model '{model_name}' is not available locally and OPENAI_API_KEY is not configured. "
            f"Pull the model with `ollama pull {model_name}` when network access is available, or set LLM_PROVIDER=openai."
        ) from exc
