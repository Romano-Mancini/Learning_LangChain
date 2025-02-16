from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool

def get_current_date(*args, **kwargs):
    """Gets the current date."""

    import datetime

    date = datetime.datetime.now()
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")

    return f"{day}/{month}/{year}"

def get_info_from_wikipedia(query: str):
    import wikipedia

    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "Could not find any information on the topic."

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
prompt = hub.pull("hwchase17/structured-chat-agent")
memory = ConversationBufferMemory()
tools = [
    Tool(
        name="Date",
        func=get_current_date,
        description="Useful to get today's date."
    ),
    Tool(
        name="Wikipedia",
        func=get_info_from_wikipedia,
        description="Useful to get information from Wikipedia."
    )
]

agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    memory=memory,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

memory.chat_memory.add_message(SystemMessage("You are an helpful chatbot that answers user's questions."))

while True:
    query = input("You: ")
    if query == 'exit':
        break
    answer = agent_executor.invoke({"input":query})
    memory.chat_memory.add_message(HumanMessage(query))
    memory.chat_memory.add_message(AIMessage(answer["output"]))

    print(f"AI: {answer['output']}")