# Contextual Compression Retriever retrieves relevant documents and then removes the unnecessary information, 
# keeping only the parts relevant to the user's question.

import os
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openrouter import ChatOpenRouter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


from config.ai_model import embedding_model

load_dotenv()

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
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
    collection_name="my_collection5",
    embedding_function=embedding_model
)


# Add documents
vector_store.add_documents(docs)

base_retriver=vector_store.as_retriever(search_kwargs={"k":5})

# setup the compressor using llm
llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)
compressor=LLMChainExtractor.from_llm(llm)

# create contextual compressor retriver
compressor_retriver=ContextualCompressionRetriever(
    base_retriever=base_retriver,
    base_compressor=compressor
)

# query
query="what is photosynthesis ?"
result=compressor_retriver.invoke(query)

for i,doc in enumerate(result):
    print(f"\n----Result{i+1}----")
    print(doc.page_content)