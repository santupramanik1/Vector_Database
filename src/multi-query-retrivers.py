# #Multi-Query Retriever generates multiple versions of a user's question and uses them to find more relevant documents.

# import chromadb
# from langchain_chroma import Chroma
# from config.ai_model import embedding_model
# from langchain_core.documents import Document
# from langchain_openrouter import ChatOpenRouter
# from langchain.retrivers.multi_query import MultiQueryRetriver
# import os
# from dotenv import load_dotenv
# load_dotenv

# all_docs = [
#     Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
#     Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
#     Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
#     Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
#     Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
#     Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
#     Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
#     Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
#     Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
#     Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
# ]


# # step2: Initialize embedding model

# # Step3: create chroma vectore store in memory
# client = chromadb.CloudClient(
#   api_key=os.getenv("CHROMA_API_KEY"),
#   tenant=os.getenv("CHROMA_TENANT"),
#   database=os.getenv("CHROMA_DATABASE")
# )

# vector_store = Chroma(
#     client=client,
#     collection_name="my_collection3",
#     embedding_function=embedding_model
# )

# vector_store.add_documents(all_docs)

# # Create retriver
# similarity_retriver=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

# multiquery_retriver=MultiQueryRetriver.from_llm(
#     retriver=vector_store.as_retriever(search_kwargs={"k":5}),
#     llm=
# )

# query="How to improve energy levels and maintain balance ?"


# # Retrive result
# similarity_result=similarity_retriver.invoke(query)
# multiquery_result=multiquery_retriver.invoke(query)

# for i,doc in enumerate(similarity_result):
#     print(f"\n----Result{i+1}----")
#     print(doc.page_content)

# print("*"*150)

# for i,doc in enumerate(multiquery_result):
#     print(f"\n----Result{i+1}----")
#     print(doc.page_content)




# Multi-Query Retriever generates multiple versions of a user's question
# and uses them to find more relevant documents.

import os
from dotenv import load_dotenv
import chromadb

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openrouter import ChatOpenRouter

from config.ai_model import embedding_model

load_dotenv()


# --------------------------------------------------
# Step 1: Sample documents
# --------------------------------------------------
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# --------------------------------------------------
# Step 2: Chroma Cloud
# --------------------------------------------------

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)


# --------------------------------------------------
# Step 3: Create vector store
# --------------------------------------------------

vector_store = Chroma(
    client=client,
    collection_name="my_collection4",
    embedding_function=embedding_model
)


# Add documents
vector_store.add_documents(all_docs)


# --------------------------------------------------
# Step 4: Normal similarity retriever
# --------------------------------------------------

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# --------------------------------------------------
# Step 5: OpenRouter LLM
# --------------------------------------------------

llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)


# --------------------------------------------------
# Step 6: Multi-Query Retriever
# --------------------------------------------------

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=llm
)


# --------------------------------------------------
# Step 7: User query
# --------------------------------------------------

query = "How to improve energy levels and maintain balance?"


# --------------------------------------------------
# Step 8: Retrieve results
# --------------------------------------------------

similarity_result = similarity_retriever.invoke(query)

multiquery_result = multiquery_retriever.invoke(query)


# --------------------------------------------------
# Step 9: Print similarity results
# --------------------------------------------------

print("\n\n========== SIMILARITY SEARCH ==========")

for i, doc in enumerate(similarity_result):
    print(f"\n---- Result {i + 1} ----")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))


# --------------------------------------------------
# Step 10: Print Multi-Query results
# --------------------------------------------------

print("\n\n========== MULTI-QUERY SEARCH ==========")

for i, doc in enumerate(multiquery_result):
    print(f"\n---- Result {i + 1} ----")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))