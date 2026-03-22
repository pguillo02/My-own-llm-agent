from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config.settings import get_settings

settings = get_settings()

def ingest(docs_path: str = settings.raw_documents_dir):
    loader = DirectoryLoader(docs_path, loader_cls = PyPDFLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = settings.chunk_size,
                                              chunk_overlap= settings.chunk_overlap)
    
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents = chunks,
        embedding = OllamaEmbeddings(model = settings.embedding_model),
        collection_name = settings.collection_name,
        persist_directory = settings.chroma_persist_dir 
    )

    print(f'Indexados {len(chunks)} chunks de {len(docs)} documentos')

if __name__ == "__main__":
    ingest()