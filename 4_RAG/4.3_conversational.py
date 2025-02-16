import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

curr_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(curr_dir, "db", "chroma_db")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm_model = ChatGroq(model="llama-3.3-70b-versatile")
llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an helpful chatbot, which answers user's questions. 
                      If you don't know the answer, reply 'I don't know the answer'.
                      {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

db = Chroma(persist_directory=db_dir, embedding_function=embedding_model)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':1, 'score_threshold': 0.5}
)

retreiver_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Given the chat history and the last user question, reformulate the question so that it can be
                      understood without any chat history. Do not answer the question, just reformulate it. Return only
                      the reformulated question."""),
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]
)

document_feeding_chain = create_stuff_documents_chain(llm_model, llm_prompt)
history_aware_retreiver = create_history_aware_retriever(llm_model, retriever, retreiver_prompt)

rag_chain = create_retrieval_chain(history_aware_retreiver, document_feeding_chain)


if __name__ == "__main__":
    chat_history = []
    while True:
        question = input("You: ")
        if question == 'exit':
            break
        answer = rag_chain.invoke({"input":question, "chat_history":chat_history})
        print(f"AI: {answer['answer']}")
        chat_history.append(HumanMessage(question))
        chat_history.append(AIMessage(answer['answer']))