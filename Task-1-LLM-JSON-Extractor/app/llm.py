"""Gemini API client wrapper for structured LLM extraction."""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from app.utils import setup_logger

# Load environment variables from .env file
load_dotenv()

logger = setup_logger("gemini_llm")


class GeminiExtractor:
    """Client wrapper for Google Gemini API structured data extraction."""

    DEFAULT_MODEL = "gemini-2.0-flash"
    FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_timeout: float = 25.0,
        client: Optional[object] = None
    ):
        """
        Initialize the Gemini Extractor.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            model_name: Optional Gemini model name. Defaults to gemini-2.0-flash or env GEMINI_MODEL.
            temperature: Sampling temperature (low for deterministic extraction).
            max_retries: Maximum number of retries upon transient errors or rate limits.
            retry_delay: Initial retry backoff delay in seconds.
            request_timeout: Request socket timeout in seconds (prevents hanging indefinitely).
            client: Optional injected client for testing.
        """
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or self.DEFAULT_MODEL
        )
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout

        if client is not None:
            self.client = client
            self.sdk_type = "custom"
        else:
            if not self.api_key:
                raise ValueError(
                    "Missing Gemini API Key. Please set the GEMINI_API_KEY or GOOGLE_API_KEY "
                    "environment variable in your .env file or environment."
                )
            self.client, self.sdk_type = self._initialize_client(self.api_key)

    def _initialize_client(self, api_key: str):
        """Initialize Google GenAI SDK client with explicit timeout."""
        try:
            from google import genai
            from google.genai import types
            http_options = types.HttpOptions(timeout=int(self.request_timeout * 1000))
            client = genai.Client(api_key=api_key, http_options=http_options)
            logger.info("Initialized Google GenAI SDK (google-genai) with %ss timeout.", self.request_timeout)
            return client, "google-genai"
        except ImportError:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=api_key)
                client = legacy_genai.GenerativeModel(self.model_name)
                logger.info("Initialized legacy Google Generative AI SDK (google-generativeai).")
                return client, "google-generativeai"
            except ImportError as err:
                raise ImportError(
                    "Neither 'google-genai' nor 'google-generativeai' is installed. "
                    "Please install dependencies via: pip install -r requirements.txt"
                ) from err

    def generate_extraction(self, prompt: str) -> str:
        """
        Send extraction prompt to the Gemini API with automatic exponential backoff retries
        and instant fallback across model candidates if a model times out or is under high demand.

        Args:
            prompt: Formatted prompt containing instructions, schema, few-shot examples, and document.

        Returns:
            Raw response text from the model.
        """
        candidate_models = [self.model_name]
        for fb in self.FALLBACK_MODELS:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_error = None
        for current_model in candidate_models:
            attempt = 0
            current_delay = self.retry_delay

            while attempt < self.max_retries:
                attempt += 1
                try:
                    logger.info("Dispatching prompt to Gemini (%s), attempt %d of %d...", current_model, attempt, self.max_retries)
                    start_time = time.time()

                    if self.sdk_type == "google-genai":
                        from google.genai import types
                        config = types.GenerateContentConfig(
                            temperature=self.temperature,
                            response_mime_type="application/json",
                            http_options=types.HttpOptions(timeout=int(self.request_timeout * 1000))
                        )
                        response = self.client.models.generate_content(
                            model=current_model,
                            contents=prompt,
                            config=config
                        )
                        raw_text = response.text or ""

                    elif self.sdk_type == "google-generativeai":
                        generation_config = {
                            "temperature": self.temperature,
                            "response_mime_type": "application/json"
                        }
                        response = self.client.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                        raw_text = response.text or ""

                    else:
                        # Custom / Mocked client for unit testing
                        if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                            res = self.client.models.generate_content(
                                model=current_model,
                                contents=prompt
                            )
                            raw_text = getattr(res, "text", str(res))
                        elif hasattr(self.client, "generate_content"):
                            res = self.client.generate_content(prompt)
                            raw_text = getattr(res, "text", str(res))
                        else:
                            raise ValueError(f"Unknown mock client interface: {type(self.client)}")

                    elapsed = time.time() - start_time
                    logger.info("Extraction completed in %.2f seconds with model %s.", elapsed, current_model)

                    if not raw_text.strip():
                        raise ValueError("Gemini returned an empty response.")

                    return raw_text

                except Exception as err:
                    last_error = err
                    err_msg = str(err)
                    is_service_demand = any(
                        code in err_msg.lower()
                        for code in [
                            "503", "504", "deadline_exceeded", "deadline expired",
                            "unavailable", "high demand", "404 not_found",
                            "no longer available", "timeout", "timed out"
                        ]
                    )
                    is_rate_limit = any(
                        code in err_msg.lower()
                        for code in ["429", "resource_exhausted", "quota", "rate limit"]
                    )

                    if is_service_demand:
                        logger.warning(
                            "Model %s encountered service delay or unavailability (%s). Immediately switching to next fallback...",
                            current_model, err_msg
                        )
                        break  # Break out immediately to next candidate model

                    if is_rate_limit and attempt < self.max_retries:
                        logger.warning("Rate limit encountered: %s. Retrying in %.1fs...", err_msg, current_delay)
                        time.sleep(current_delay)
                        current_delay *= 2.0
                    else:
                        logger.error("Gemini API call failed with model %s: %s", current_model, err_msg)
                        raise err

        raise RuntimeError(f"Extraction failed across available models. Last error: {last_error}")
