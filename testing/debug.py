import os
from dotenv import load_dotenv
load_dotenv()

print("=== Variables de entorno ===")
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY", "NO ENCONTRADA")[:12], "...")
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
print("LANGCHAIN_ENDPOINT:", os.getenv("LANGCHAIN_ENDPOINT"))

print("\n=== Test conexión LangSmith ===")
from langsmith import Client

try:
    client = Client()
    projects = list(client.list_projects())
    print("Conexión OK. Proyectos existentes:")
    for p in projects:
        print(f"  - {p.name}")
except Exception as e:
    print("Error de conexión:", e)