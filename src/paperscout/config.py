from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from paperscout.models.endpoints import normalize_local_base_url


class Settings(BaseSettings):
    """Application settings shared by the CLI and later agent components."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPERSCOUT_",
        extra="ignore",
        case_sensitive=False,
    )

    environment: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    model_base_url: str = "http://127.0.0.1:8001/v1"
    model_api_key: str = "EMPTY"
    model_name: str = "Qwen/Qwen3-8B"
    model_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    model_max_tokens: int = Field(default=2048, gt=0, le=32768)
    model_temperature: float = Field(default=0.2, ge=0, le=2)
    use_model_reasoning: bool = False

    retrieval_mode: str = "lexical"
    embedding_model: str = "BAAI/bge-m3"
    vector_index_path: Path = Path("data/vector.index")
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = False

    max_steps: int = Field(default=12, gt=0, le=100)
    max_tool_calls: int = Field(default=24, gt=0, le=200)
    max_papers: int = Field(default=20, gt=0, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("model_base_url")
    @classmethod
    def normalize_model_base_url(cls, value: str) -> str:
        return normalize_local_base_url(value)

    def prepare_directories(self, root: Path | None = None) -> None:
        """Create configured storage directories relative to ``root``."""
        base_dir = root or Path.cwd()
        for directory in (self.data_dir, self.runs_dir):
            path = directory if directory.is_absolute() else base_dir / directory
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
