from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
import asyncio
import os

def get_llm():
    azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE") or ""
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION") or ""
    azure_openai_deployment : str = os.getenv("AZURE_OPENAI_MODEL") or ""
    llm = AzureChatOpenAI(azure_deployment=azure_openai_deployment, temperature=0, streaming=False, 
                          azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
    return llm

def get_graph():
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        refresh_schema=False
    )
    return graph

def clean_graph(graph):
    input("Press Enter to delete graph...")
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.query(query)

text = """
Marie Curie, 7 November 1867 – 4 July 1934, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
Also, Robin Williams!
"""
documents = [Document(page_content=text)]
llm = get_llm()
graph = get_graph()

async def process_graph_no_schema():
    transformer = LLMGraphTransformer(llm=llm)
    data = await transformer.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

async def process_graph_allowed_nodes():
    allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
    transformer = LLMGraphTransformer(llm=llm, allowed_nodes=allowed_nodes)
    data = await transformer.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

async def process_graph_allowed_nodes_relations_1():
    allowed_nodes = ["Person", "Organization", "Place", "Award", "ResearchField"]
    allowed_relationships = ["SPOUSE", "AWARD", "FIELD_OF_RESEARCH", "WORKS_AT", "IN_LOCATION"]
    transfomer = LLMGraphTransformer(llm=llm,allowed_nodes=allowed_nodes,allowed_relationships=allowed_relationships)
    data = await transfomer.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

async def process_graph_allowed_nodes_relations_2():
    allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
    allowed_relationships = [
        ("Person", "SPOUSE", "Person"),
        ("Person", "AWARD", "Award"),
        ("Person", "WORKS_AT", "Organization"),
        ("Organization", "IN_LOCATION", "Location"),
        ("Person", "FIELD_OF_RESEARCH", "ResearchField")
    ]
    transformer = LLMGraphTransformer(llm=llm, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships)
    data = await transformer.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

async def process_graph_allowed_nodes_relations_properties_1():
    allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
    allowed_relationships = [
        ("Person", "SPOUSE", "Person"),
        ("Person", "AWARD", "Award"),
        ("Person", "WORKS_AT", "Organization"),
        ("Organization", "IN_LOCATION", "Location"),
        ("Person", "FIELD_OF_RESEARCH", "ResearchField")
    ]
    node_properties=True
    relationship_properties=True
    t = LLMGraphTransformer(llm=llm, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships, 
        node_properties=node_properties,
        relationship_properties=relationship_properties
    )
    data = await t.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

async def process_graph_allowed_nodes_relations_properties_2():
    allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
    allowed_relationships = [
        ("Person", "SPOUSE", "Person"),
        ("Person", "AWARD", "Award"),
        ("Person", "WORKS_AT", "Organization"),
        ("Organization", "IN_LOCATION", "Location"),
        ("Person", "FIELD_OF_RESEARCH", "ResearchField")
    ]
    node_properties=["birth_date", "death_date"]
    relationship_properties=["start_date"]
    props_defined = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    node_properties=node_properties,
    relationship_properties=relationship_properties
    )
    data = await props_defined.aconvert_to_graph_documents(documents)
    print(data)
    graph.add_graph_documents(data)

clean_graph(graph)
asyncio.run(process_graph_no_schema())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes_relations_1())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes_relations_2())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes_relations_properties_1())

clean_graph(graph)
asyncio.run(process_graph_allowed_nodes_relations_properties_2())

clean_graph(graph)