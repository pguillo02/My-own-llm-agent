from config.settings import get_settings
settings = get_settings()

from rag.chain import build_rag_chain
from evaluation.dataset import DatasetManager
from evaluation.evaluators import RagasEvaluator

def main():
    manager = DatasetManager()
    manager.load()

    chain = build_rag_chain()
    ragas_dataset = manager.as_ragas_dataset(chain)

    evaluator = RagasEvaluator(ragas_dataset, "eval-v1")

    results = evaluator.run_evaluation()

    df = results.to_pandas()
    df.to_csv("./data/eval_results/eval-v1.csv", index=False)
    print("Evaluation completed")

if __name__ == "__main__":
    main()