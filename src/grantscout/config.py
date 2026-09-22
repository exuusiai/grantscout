from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from grantscout.models.endpoints import normalize_local_base_url, normalize_research_base_url


class Settings(BaseSettings):
    """Application settings shared by the CLI and later agent components."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GRANTSCOUT_",
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
    # Optional separate local OpenAI-compatible endpoint for paper analysis.
    research_model_base_url: str | None = None
    research_model_api_key: str = "EMPTY"
    research_model_name: str | None = None

    retrieval_mode: str = "lexical"
    embedding_model: str = "BAAI/bge-m3"
    vector_index_path: Path = Path("data/vector.index")
    # Torch device for the local embedding/reranker models (e.g. "cuda:1") so they
    # can stay off a GPU reserved for another workload such as vLLM.
    embedding_device: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None
    use_reranker: bool = False
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_timeout_seconds: float = Field(default=12.0, gt=0, le=60)
    arxiv_max_results: int = Field(default=20, gt=0, le=100)

    max_steps: int = Field(default=12, gt=0, le=100)
    max_tool_calls: int = Field(default=24, gt=0, le=200)
    max_papers: int = Field(default=20, gt=0, le=1000)
    max_retries: int = Field(default=2, ge=0, le=10)
    # Proposal drafting: expansion passes per section to approach word budgets.
    proposal_expansion_passes: int = Field(default=2, ge=0, le=5)
    # Optional external file server for citation jump links (pdf.js viewer base).
    # Citation URL convention: {base}?file={source_path}#page={page}
    file_server_base_url: str = ""
    # PII scrubbing: private-domain documents are scrubbed before entering the index.
    pii_scrub_private: bool = True
    pii_presidio: bool = True
    pii_extra_words: str = ""
    pii_extra_words: str = ""
    # /web 学术搜索:OpenAlex 检索 + 可选外部模型综合(仅公开元数据外发)。
    web_api_base_url: str = ""
    web_api_key: str = ""
    web_api_model: str = ""
    # Prometheus 指标(/metrics),默认关闭。
    metrics_enabled: bool = False

    @field_validator("model_base_url")
    @classmethod
    def normalize_model_base_url(cls, value: str) -> str:
        return normalize_local_base_url(value)

    @field_validator("research_model_base_url")
    @classmethod
    def normalize_research_model_base_url(cls, value: str | None) -> str | None:
        return normalize_research_base_url(value) if value else None

    def prepare_directories(self, root: Path | None = None) -> None:
        """Create configured storage directories relative to ``root``."""
        base_dir = root or Path.cwd()
        for directory in (self.data_dir, self.runs_dir):
            path = directory if directory.is_absolute() else base_dir / directory
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
