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

def get_ragas_config():
    """
    
    """

    llm = LangchainLLMWrapper(
        ChatOllama(model = settings.llm_model, temperature = 0)
    )

    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model = settings.embedding_model)
    )

    return llm, embeddings

def run_evaluation(ragas_dataset, experiment_name: str = "eval-v1") -> dict:
    """
    
    """

    llm, embeddings = get_ragas_config()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        answer_correctness
    ]

    for metric in metrics:
        metric.llm = llm

        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print(f"Executing validation '{experiment_name}'...")
    print(f"Examples: {len(ragas_dataset)}")
    print(f"Metrics: {[m.name for m in metrics]}")

    results = evaluate(
        dataset = ragas_dataset,
        metrics = metrics,
    )

    scores = results.to_pandas()
    print("\n=== Results ===")
    
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]:
        if metric in scores.columns:
            mean = scores[metric].mean()
            print(f" {metric}: {mean: .3f}")
    
    return results
