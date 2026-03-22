from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):

    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str
    langchain_project: str = "My-own-llm-agent"

    chroma_persist_dir: str = './data/chroma'
    raw_documents_dir: str = './data/docs'
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4

    embedding_model: str = "mxbai-embed-large"
    llm_model: str = "llama3.2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()

if __name__ == "__main__":
    print(get_settings())
