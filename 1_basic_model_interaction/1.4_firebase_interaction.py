from dotenv import load_dotenv
from os import getenv
from langchain_groq import ChatGroq
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

PROJECT_ID = getenv("PROJECT_ID")
SESSION_ID = "test"
COLLECTION_NAME = "chat_history"

client = firestore.Client(project=PROJECT_ID)

model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)


while True:
    user_message = input("You: ")
    if user_message == "exit":
        break

    chat_history.add_user_message(user_message)

    ai_message = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_message.content)
    print(f"AI: {ai_message.content}")

for index, message in enumerate(chat_history.messages):
    print(f"{index}) {message.content}")