import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_directory = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_directory, "db", "chroma_db")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=db_dir,
    embedding_function=embedding_model
)

retreiver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':1, 'score_threshold': 0.2}
)

documents_retreived = retreiver.invoke("How much money does entering a new F1 team require?")
for doc in documents_retreived:
    print(f"Source: {doc.metadata.get('source', 'unknown')}, Content: {doc.page_content}")
    print("-------------------")