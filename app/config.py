"""Configuration settings for The Relay."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PAPERS_DIR: Path = DATA_DIR / "papers"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    RESULTS_DIR: Path = BASE_DIR / "results"
    RESULTS_CSV: Path = RESULTS_DIR / "results.csv"
    BENCHMARK_CONFIG: Path = RESULTS_DIR / "benchmark_config.json"
    QUESTIONS_FILE: Path = DATA_DIR / "questions.jsonl"

    # Groq API configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Model provider selection: "simulator" or "groq"
    MODEL_PROVIDER: str = "simulator"
    SIMULATOR_URL: str = "http://127.0.0.1:8001"

    # Rate Limiting & Admission Control
    RATE_LIMIT_RPM: int = 30
    RATE_LIMIT_TPM: int = 6000
    DAILY_REQUEST_BUDGET: int = 1000
    DAILY_TOKEN_BUDGET: int = 100000
    MAX_CONCURRENCY: int = 5
    QUEUE_CAPACITY: int = 20
    QUEUE_TIMEOUT_SECONDS: float = 10.0

    # Retry Policy
    MAX_RETRIES: int = 1
    RETRY_BASE_BACKOFF_SECONDS: float = 0.5
    RETRY_MAX_BACKOFF_SECONDS: float = 3.0

    # Retrieval
    DEFAULT_RETRIEVAL_SETTING: str = "B"
    CHUNKING_STRATEGY: str = "sentence"
    SIMILARITY_THRESHOLD: float = 0.07

    # Verification threshold
    MIN_CITATION_OVERLAP_RATIO: float = 0.15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
