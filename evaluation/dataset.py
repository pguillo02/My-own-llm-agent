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
