from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import os
from dotenv import load_dotenv
from config.ai_model import embedding_model

load_dotenv()

#Step1: Source document
documents=[
    Document(page_content="Langchain helps developer to build LLm application easily"),
    Document(page_content="Chroma is vectore database optimized for LLM based search"),
    Document(page_content="Embedding convert text int high-dimensional vectors"),
    Document(page_content="OpenAi provides powerfull embedding models")
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
    collection_name="my_collection2",
    embedding_function=embedding_model
)

vector_store.add_documents(documents)

# Step4: convert vectorstores into retrivers
retriver=vector_store.as_retriever(search_kwargs={"k":2})

query="what is chroma used for?"
result=retriver.invoke(query)

print(result)

for i,doc in enumerate(result):
    print(f"\n----Result{i+1}-----")
    print(doc.page_content)



