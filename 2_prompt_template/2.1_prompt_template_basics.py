from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

print("----- Simple Template -----")
# Simple template
template = "Tell me a joke about {subject}. Make it {lenght} sentences long."
template = ChatPromptTemplate.from_template(template)
human_message = template.invoke({"subject":"cats", "lenght":"three"})
print(model.invoke(human_message).content)

print("----- Message Template -----")
# Message template, use tuples!
messages = [
    ("system", "You are a comedian that tells jokes on demand about {topic}."),
    ("human", "Tell me {number} jokes.")
]

prompt = ChatPromptTemplate.from_messages(messages)
print(model.invoke(prompt.invoke({"topic":"computer science", "number":"two"})).content)