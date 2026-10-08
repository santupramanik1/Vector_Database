from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_openrouter import ChatOpenRouter
from config.ai_model import embedding_model
import os 
from dotenv import load_dotenv
load_dotenv()

# tool create
@tool
def multiply(a:int,b:int)->int:
    """ given two number a and b this tool return their products"""
    return a*b

# tool binding
llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)

llm_with_tools=llm.bind_tools([multiply])

query=HumanMessage("can you multiplay 3 and 1000")

message=[query]

result=llm_with_tools.invoke(message)

message.append(result)

tool_result=multiply.invoke(result.tool_calls[0])

message.append(tool_result)


# print(message)


ans=llm_with_tools.invoke(message).content

print(ans)