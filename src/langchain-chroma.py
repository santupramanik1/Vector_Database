import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from config.ai_model import connection

load_dotenv()

#create langchain document for IPL player
doc1 = Document(
    page_content="""
    Virat Kohli is an Indian cricketer and right-handed batsman.
    He is known for his consistency, aggressive batting,
    and leadership skills. He has been a key player in IPL cricket.
    """,
    metadata={
        "team": "Royal Challengers Bengaluru"
    }
)

doc2 = Document(
    page_content="""
    MS Dhoni is an Indian cricketer, wicketkeeper, and batsman.
    He is known for his calm leadership, finishing ability,
    and experience in IPL cricket. He has captained Chennai
    Super Kings for many seasons.
    """,
    metadata={
        "team": "Chennai Super Kings"
    }
)

doc3 = Document(
    page_content="""
    Rohit Sharma is an Indian cricketer and right-handed batsman.
    He is known for his powerful hitting, leadership,
    and excellent batting skills. He has played a significant
    role in Mumbai Indians' IPL success.
    """,
    metadata={
        "team": "Mumbai Indians"
    }
)

doc4 = Document(
    page_content="""
    Jasprit Bumrah is an Indian fast bowler known for his
    unique bowling action, accurate yorkers, and death-over
    bowling skills. He is one of the key bowlers in IPL cricket.
    """,
    metadata={
        "team": "Mumbai Indians"
    }
)

doc5 = Document(
    page_content="""
    Sanju Samson is an Indian wicketkeeper-batsman known for
    his powerful stroke play and excellent batting skills.
    He has served as a captain and key player in the IPL.
    """,
    metadata={
        "team": "Rajasthan Royals"
    }
)

docs=[doc1,doc2,doc3,doc4,doc5]

client = chromadb.CloudClient(
  api_key=os.getenv("CHROMA_API_KEY"),
  tenant=os.getenv("CHROMA_TENANT"),
  database=os.getenv("CHROMA_DATABASE")
)

vector_store = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=connection
)

doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
vector_store.add_documents(docs, ids=doc_ids)

embeddings = connection.embed_documents([doc.page_content for doc in docs])
print(embeddings)

results = vector_store.similarity_search(
    query="Who is a wicketkeeper-batsman known for finishing games?",
    k=1
)

for res in results:
    print(f"* {res.page_content.strip()} [{res.metadata}]")



