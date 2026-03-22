from langchain_chroma import Chroma 
from langchain_core.vectorstores import VectorStoreRetriever
from config.settings import get_settings
from langchain_ollama import OllamaEmbeddings

settings = get_settings()

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name = settings.collection_name,
        embedding_function = OllamaEmbeddings(model = settings.embedding_model),
        persist_directory = settings.chroma_persist_dir
    )