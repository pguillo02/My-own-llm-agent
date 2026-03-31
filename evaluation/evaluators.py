from ragas import evaluate 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import Client 
from config.settings import get_settings
from langchain_ollama import ChatOllama, OllamaEmbeddings

settings = get_settings()

class RagasEvaluator():

    def __init__(self, ragas_dataset, experiment_name):
        self.ragas_dataset = ragas_dataset
        self.experiment_name = experiment_name
        self.ragas_llm = ChatOllama(model = settings.llm_model, temperature = 0)
        self.ragas_embedded = OllamaEmbeddings(model = settings.embedding_model)
        self.metrics = settings.metrics

    def run_evaluation(self) -> dict:
        """
    
        """

        for metric in self.metrics:
            metric.llm = self.ragas_llm

            if hasattr(metric, "embeddings"):
                metric.embeddings = self.ragas_embedded

        print(f"Executing validation '{self.experiment_name}'...")
        print(f"Examples: {len(self.ragas_dataset)}")
        print(f"Metrics: {[m.name for m in self.metrics]}")

        results = evaluate(
            dataset = self.ragas_dataset,
            metrics = self.metrics,
        )

        scores = results.to_pandas()
        print("\n=== Results ===")
    
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]:
            if metric in scores.columns:
                mean = scores[metric].mean()
                print(f" {metric}: {mean: .3f}")
    
        return results
