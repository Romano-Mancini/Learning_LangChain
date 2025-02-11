from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a college professor that is holding a lecture about {topic}. You answer questions from the crowd."),
        ('human', "Are there any business opportunities in the {field} field?")
    ]
)

lowercase = RunnableLambda(lambda x: x.lower())
word_counter = RunnableLambda(lambda x: len(set(x.split()))) 
chain = prompt | model | StrOutputParser() | lowercase | word_counter

print(chain.invoke({"topic":"LLM", "field":"tourism"}))