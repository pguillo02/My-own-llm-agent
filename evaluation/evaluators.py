from ragas import evaluate 
from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    answer_correctness
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import Client 
from config.settings import get_settings

settings = get_settings()

