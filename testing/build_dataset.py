from dotenv import load_dotenv
load_dotenv()

from config.settings import get_settings
get_settings()

from evaluation.dataset import DatasetManager
from rag.chain import build_rag_chain

def main():
    manager = DatasetManager()

    manager.load()

    manager.push_to_langsmith()

if __name__ == "__main__":
    main()