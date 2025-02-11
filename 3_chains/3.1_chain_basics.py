from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a comedian that tells jokes on request about {topic}'),
        ('human', 'Tell me {number} jokes.')
    ]
)

chain = template | model | StrOutputParser()

print(chain.invoke({'topic':'politics', 'number':'three'}))