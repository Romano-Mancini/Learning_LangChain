import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

curr_dir = os.path.dirname(os.path.abspath(__file__))
article_dir = os.path.join(curr_dir, "wikipedia_articles", "formula1.txt")
db_dir = os.path.join(curr_dir, "db", "chroma_db")

print(f"""Current directory: {curr_dir}
Article directory: {article_dir}
Vector Store directory: {db_dir}""")

if not os.path.exists(db_dir):
    
    if not os.path.exists(article_dir):
        raise FileNotFoundError(f"The path {article_dir} does not exist.")

    loader = TextLoader(article_dir)
    document = loader.load()
    for doc in document:
        doc.metadata={"source":os.path.basename(article_dir)}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitted_documents = text_splitter.split_documents(document)

    print(f"Number of splitted documents: {len(splitted_documents)}.")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=splitted_documents,
        embedding=embedding_model,
        persist_directory=db_dir
    )

    print(len(db.get()['documents']))