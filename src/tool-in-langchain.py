# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.tools import ShellTool
# from langchain_community.tools import tool
# Web search
# search_tool=DuckDuckGoSearchRun()
# result=search_tool.invoke("Today's important news in india")
# print(result)

# shell tool
# shell_tool=ShellTool()
# ans=shell_tool.invoke("whoami")
# print(ans)

# Custom tools

# #step 1: create a function
# def multiply(a,b):
#     """multiply two numbers"""
#     return a*b

# # step 2: add type  hints 
# def multiply(a:int,b:int)->int:
#     """multiply two numbers"""
#     return a*b

# # step3 :add tool decorator
# @tool
# def multiply(a:int,b:int)->int:
#     """multiply two numbers"""
#     return a*b

# result=multiply.invoke({"a":3,"b":5})
# print(result)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)


from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field

class MultiplyInput(BaseModel):
    a:int=Field(required=True,description="first number to add")
    b:int=Field(required=True,description="second number to add")


def multiply_func(a:int,b:int)->int:
    return a*b

multiply_tool=StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="multiply two numbers",
    args_schema=MultiplyInput
)

result=multiply_tool.invoke({'a':3,'b':5})

print(result)

