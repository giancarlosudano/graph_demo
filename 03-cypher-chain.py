from langchain_openai import AzureChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import os

def get_llm():
    azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE") or ""
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION") or ""
    azure_openai_deployment : str = os.getenv("AZURE_OPENAI_MODEL") or ""
    llm = AzureChatOpenAI(azure_deployment=azure_openai_deployment, temperature=0, streaming=False, 
                          azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
    return llm

url=os.getenv("ICIJ_NEO4J_URL")
username=os.getenv("ICIJ_NEO4J_USERNAME")
password=os.getenv("ICIJ_NEO4J_PASSWORD")

graph = Neo4jGraph(url=url, username=username, password=password)
llm = get_llm()
chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)

chain.invoke({'query': "Which intermediary is connected to most entites?"})
input("Press Enter to continue...")

chain.invoke({'query':"Who are the officers of ZZZ-MILI COMPANY LTD.?"})
input("Press Enter to continue...")

chain.invoke({'query':
"""How are entities SOUTHWEST LAND DEVELOPMENT LTD. and Dragon Capital Markets Limited connected?
Find a shortest path."""})
input("Press Enter to continue...")

# https://towardsdatascience.com/langchain-has-added-cypher-search-cb9d821120d5
# https://github.com/tomasonjo/blogs/blob/master/llm/langchain_neo4j.ipynb
# https://sandbox.neo4j.com/?usecase=icij-paradise-papers