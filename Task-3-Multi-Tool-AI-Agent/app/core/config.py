"""
Configuration management for Task 3: Multi-Tool Autonomous AI Agent.
Safely loads environment variables, validates paths, and provides defaults.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory of the Task 3 project
TASK3_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file from the Task 3 directory
load_dotenv(TASK3_DIR / ".env")


def get_masked_api_key(key: Optional[str]) -> str:
    """Return a masked representation of an API key for safe logging."""
    if not key:
        return "[NOT SET]"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class AppConfig:
    """Application configuration container."""

    # Base paths
    task_root: Path = field(default_factory=lambda: TASK3_DIR)
    data_dir: Path = field(
        default_factory=lambda: (
            TASK3_DIR / os.getenv("DATA_DIR", "data")
        ).resolve()
    )

    # Gemini LLM Settings
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
    )
    gemini_temperature: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
    )

    # Search Settings
    search_provider: str = field(
        default_factory=lambda: os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    )
    search_max_results: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    )
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # Execution Bounds
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )
    max_file_size_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
        )
    )

    def validate(self) -> None:
        """Validate critical configuration paths."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global singleton configuration instance
config = AppConfig()
config.validate()
