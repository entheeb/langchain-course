import os
from typing import List

from dotenv import load_dotenv
from langchain import tools
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field


load_dotenv()  # Load environment variables from .env file

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


def main():
    print("Hello from langchain-course!")
    print(f"Model Name: {os.getenv('MODEL_NAME')}")

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
    tools = [search]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Germany on linkedin and list their details")})
    print(type(result))

if __name__ == "__main__":
    main()
