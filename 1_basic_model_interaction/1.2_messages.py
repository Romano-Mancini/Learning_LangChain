from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

messages = [
    SystemMessage(content="Respond to math questions in ancient English."),
    HumanMessage(content="What is the square root of 49?")
]

response = model.invoke(messages)
print(f"Response: {response.content}")