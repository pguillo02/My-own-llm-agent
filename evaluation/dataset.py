import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from langsmith import Client
from datasets import Dataset
from config.settings import get_settings

settings = get_settings()

@dataclass
class Example: 
    id: str
    type: str 
    question: str 
    ground_truth: str
    metadata: dict

class DatasetManager:
    def __init__(self, json_path: str = '.data/datasets/golden_set.json'):
        self.json_path = Path(json_path)
        self.client = Client()
        self._examples = list[Example] = []

    def load(self) -> list[Example]:
        """
        
        """

        with open(self.json_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)

        self._examples = [
            Example(
                id = ex["id"],
                type = ex["type"],
                question = ex["question"],
                ground_truth = ex["ground_truth"],
                metadata = ex["metadata", {}]
            ) 
            for ex in data["examples"]
        ]

        print(f"Loaded {len(self._examples)} example questions")
        print(f"  - Factual:   {sum(1 for e in self._examples if e.type == 'factual')}")
        print(f"  - Partial:   {sum(1 for e in self._examples if e.type == 'partial')}")
        print(f"  - No answer: {sum(1 for e in self._examples if e.type == 'no_answer')}")

        return self._examples

    def push_to_langsmith(self, dataset_name: Optional[str] = None) -> str:
        """
        
        """

        if not self._examples:
            self.load()

        name = dataset_name or self.json_path.stem

        existing = [d for d in self.client.list_datasets() if d.name == name]

        if existing:
            dataset = existing[0]

            print(f"Dataset '{name}' already registered on LangSmith, updating...")

        else:
            dataset = self.client.create_dataset(
                dataset_name = name,
                description = f"Golden set - {len(self._examples)} examples"
            )

            print(f"Dataset '{name}' created on LangSmith")

        self.client.create_examples(
            inputs = [{"question": e.question} for e in self._examples],
            outputs = [{"ground_truth": e.ground_truth} for e in self._examples],
            metadata = [e.metadata for e in self._examples],
            dataset_id = dataset.id
        )

        print(f"{len(self._examples)} uploaded to LangSmith")
        return dataset.id

        


