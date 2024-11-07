import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.document_loaders import WikipediaLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph

def clean_graph(graph):
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        refresh_schema=False
    )
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph.query(query)

# Read the wikipedia article
raw_documents = WikipediaLoader(query="The Witcher").load()
# Define chunking strategy
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=20
)

# Chunk the document
documents = text_splitter.split_documents(raw_documents)
for d in documents:
    del d.metadata["summary"]

url=os.getenv("NEO4J_URL")
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")
azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version: str = os.getenv("AZURE_OPENAI_API_VERSION")
azure_embedding_deployment : str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

neo4j_db = Neo4jVector.from_documents(
    documents,
    AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, api_key=api_key, azure_deployment=azure_embedding_deployment),
    url=url,
    username=username,
    password=password,
    database="neo4j",  # neo4j by default
    index_name="wikipedia",  # vector by default
    node_label="WikipediaArticle",  # Chunk by default
    text_node_property="info",  # text by default
    embedding_node_property="vector",  # embedding by default
    create_id_index=True,  # True by default
)

neo4j_db.query("SHOW CONSTRAINTS")

neo4j_db.query(
    """SHOW INDEXES
       YIELD name, type, labelsOrTypes, properties, options
       WHERE type = 'VECTOR'
    """
)

neo4j_db.add_documents(
    [
        Document(
            page_content="LangChain is the coolest library since the Library of Alexandria",
            metadata={"author": "Tomaz", "confidence": 1.0}
        )
    ],
    ids=["langchain"],
)

existing_index = Neo4jVector.from_existing_index(
    AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, api_key=api_key, azure_deployment=azure_embedding_deployment),
    url=url,
    username=username,
    password=password,
    index_name="wikipedia",
    text_node_property="info",  # Need to define if it is not default
)

# Learn how to customize LangChain’s wrapper of Neo4j vector index
# blog https://blog.langchain.dev/neo4j-x-langchain-new-vector-index/
# notebook https://github.com/tomasonjo/blogs/blob/master/llm/neo4jvector_langchain_deepdive.ipynb
