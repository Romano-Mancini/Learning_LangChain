from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

results = model.invoke("What is the square root of 49?")
print(f"Full Results: {results}")
print(f"Content Results: {results.content}")