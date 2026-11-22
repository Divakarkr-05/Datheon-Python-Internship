"""
LLM initialization for Task 3: Multi-Tool Autonomous AI Agent.
Initializes Google Gemini with REST transport to ensure reliability on Windows systems.
"""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import config, get_masked_api_key
from app.core.logger import logger


def get_gemini_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatGoogleGenerativeAI:
    """
    Initialize and return a ChatGoogleGenerativeAI LLM instance.
    Uses transport='rest' for stable execution on Windows 11 under Application Control.
    """
    effective_api_key = api_key or config.gemini_api_key
    if not effective_api_key:
        raise ValueError(
            "Google Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    effective_model = model_name or config.gemini_model
    effective_temp = temperature if temperature is not None else config.gemini_temperature

    logger.debug(
        "Initializing Gemini LLM: model=%s, temp=%s, key=%s",
        effective_model,
        effective_temp,
        get_masked_api_key(effective_api_key),
    )

    return ChatGoogleGenerativeAI(
        model=effective_model,
        temperature=effective_temp,
        google_api_key=effective_api_key,
        transport="rest",
        max_retries=2,
        request_timeout=30.0,
    )
