from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

load_dotenv()

def get_current_date(*args, **kwargs):
    """Gets the current time."""

    import datetime

    date = datetime.datetime.now()
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")

    return f"{day}/{month}/{year}"

tools = [
    Tool(
        name="date",
        func=get_current_date,
        description="Useful to get the current date."
    )
]

prompt = hub.pull("hwchase17/react")

model = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

answer = agent_executor.invoke({"input":"What is the date of today?"})

print(f"Final answer: {answer['output']}")