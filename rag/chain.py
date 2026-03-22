from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from config.settings import get_settings
from rag.retriever import get_retriever
from rag.prompts import RAG_PROMPT

settings = get_settings()

