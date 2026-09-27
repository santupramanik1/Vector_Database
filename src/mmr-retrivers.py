# MMR (Maximal Marginal Relevance) is a retrieval method that selects documents 
# that are both relevant to the query and different from each other.

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.ai_model import embedding_model
import os
from dotenv import load_dotenv

load_dotenv()
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more.")
]

# step2: Initialize embedding model

# Step3: create chroma vectore store in memory
client = chromadb.CloudClient(
  api_key=os.getenv("CHROMA_API_KEY"),
  tenant=os.getenv("CHROMA_TENANT"),
  database=os.getenv("CHROMA_DATABASE")
)

vector_store = Chroma(
    client=client,
    collection_name="my_collection3",
    embedding_function=embedding_model
)

vector_store.add_documents(docs)

# Enable MMR to retriver
retriver=vector_store.as_retriever(search_type="mmr", #<-- this enable mmr
                                   search_kwargs={"k":3,"lambda_mult":0.5}) # k=top result, lambda_mult = relavance diverse result

query="what is langchain ?"
result=retriver.invoke(query)

for i,doc in enumerate(result):
    print(f"\n----Result{i+1}----")
    print(doc.page_content)


