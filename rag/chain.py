from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from config.settings import get_settings
from rag.retriever import get_retriever
from rag.prompts import RAG_PROMPT

settings = get_settings()

def format_docs(docs: list) -> str:
    """
    Function that formats the retrieved documents by indicating its sources and concatenates them with two lines as a separator.

    Input: a list of documents. 

    Returns: the documents concatanated with its sources. 
    """

    return "/n/n".join([f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}" for doc in docs])

def build_rag_chain():
    """ 
    
    """

    retriever = get_retriever()
    llm = ChatOllama(model = settings.llm_model)

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | RAG_PROMPT
             | llm 
             | StrOutputParser()
    )

    return chain
    