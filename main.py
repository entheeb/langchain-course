import os

from dotenv import load_dotenv
from langchain import tools
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient


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


def main():
    print("Hello from langchain-course!")
    print(f"Model Name: {os.getenv('MODEL_NAME')}")

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
    tools = [search]
    agent = create_agent(model=llm,tools=tools)

    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()
