import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from langsmith import Client
from datasets import Dataset
from config.settings import get_settings

settings = get_settings()