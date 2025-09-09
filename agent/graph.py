from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.prompts import *
from agent.states import *

_ = load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

class Plan(BaseModel):
    pass


# nodes
def planner_agent(state: dict) -> dict:
    prompt = state['user_prompt']
    resp = llm.with_structured_output(Plan).invoke(prompt)
    return { "plan": resp }


# graph
graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.set_entry_point("planner")

agent = graph.compile()

user_prompt = "create a simple calculator web application"

result = agent.invoke({"user_prompt": user_prompt})
print(result[END])