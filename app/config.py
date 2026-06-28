from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:4b", alias="OLLAMA_MODEL")
    embedding_model: str = Field(default="bge-m3", alias="EMBEDDING_MODEL")
    docs_dir: Path = Field(default=Path("data/docs"), alias="DOCS_DIR")
    index_dir: Path = Field(default=Path("storage/index"), alias="INDEX_DIR")
    cards_file: Path = Field(default=Path("data/cards/cards.json"), alias="CARDS_FILE")
    top_k: int = Field(default=4, alias="TOP_K")
    candidate_top_k: int = Field(default=30, alias="CANDIDATE_TOP_K")
    lexical_top_k: int = Field(default=30, alias="LEXICAL_TOP_K")
    max_evidence: int = Field(default=10, alias="MAX_EVIDENCE")
    rebuild_index: bool = Field(default=False, alias="REBUILD_INDEX")


@lru_cache
def get_settings() -> Settings:
    return Settings()
