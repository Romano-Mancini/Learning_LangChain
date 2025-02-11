from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage 

load_dotenv()

chat_history = []

model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history.append(SystemMessage("Respond to math question, and explain briefly the answers."))

while True:
    question = input("You: ")
    if question == "exit":
        break

    chat_history.append(HumanMessage(question))
    response = model.invoke(chat_history)
    print(f"AI: {response.content}")
    chat_history.append(AIMessage(response.content))

for index, element in enumerate(chat_history):
    print(f"{index}) {element.content}")