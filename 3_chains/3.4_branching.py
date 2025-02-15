from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

positive_review = "The product is eccellent, I love it!"
negative_review = "The product is awful, and its quality is orrible."
neutral_review = "I still got to try the product."

model = ChatGroq(model="llama-3.3-70b-versatile")

start_template = ChatPromptTemplate.from_messages([
    ('system', """You are the head of a team that received a feedback from the user.
                 You will respond with "positive", "negative" or "neutral" depending on the sentiment of the feedback.
                  Do not output anything other than the prvided words."""),
    ('human', "{review}")
])

positive_template = ChatPromptTemplate.from_messages([
    ('system', """You are the head of a team that received a good feedback from the user.
                  You will respond sweetly."""),
    ('human', "{review}")
])

negative_template = ChatPromptTemplate.from_messages([
    ('system', """You are the head of a team that received a bad feedback from the user.
                  You will respond asking politely for more details."""),
    ('human', "{review}")
])

neutral_template = ChatPromptTemplate.from_messages([
    ('system', """You are the head of a team that received a neutral feedback from the user.
                  You will thank the user for their time."""),
    ('human', "{review}")
])

default_template = ChatPromptTemplate.from_messages([
    ('system', """You are the head of a team that wants to receive feedback for their product.
                  You will ask for a review."""),
    ('human', "{review}")
])

branches = RunnableBranch(
    (
        lambda x : "positive" in x,
        positive_template | model | StrOutputParser()
    ),
    (
        lambda x : "negative" in x,
        negative_template | model | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x,
        neutral_template | model | StrOutputParser()
    ),
    default_template | model | StrOutputParser()
)

final_chain = start_template | model | StrOutputParser() | branches

print("----Positive review----")
print(final_chain.invoke({"review":positive_review}))
print("----Negative review----")
print(final_chain.invoke({"review":negative_review}))
print("----Neutral review----")
print(final_chain.invoke({"review":neutral_review}))