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

    
