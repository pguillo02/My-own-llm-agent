from dotenv import load_dotenv
load_dotenv() 

from rag.chain import build_rag_chain, query_rag

def main():
    chain = build_rag_chain()

    questions = [
        "¿Qué experiencia profesional tiene Pablo Guilló?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = query_rag(question, chain=chain)
        print(f"A: {result['answer']}")
        print(f"Fuentes: {[d['source'] for d in result['source_documents']]}")

if __name__ == "__main__":
    main()