from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import chromadb
from dotenv import load_dotenv
import os
from src.config.ai_model import embedding_model
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Step 1: Document ingestion (Indexing):
video_id = "Gfr50f6ZBvo" 
try:
    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id, languages=["en"])

    transcript_list = transcript.to_raw_data()

    # print(transcript_list)

    transcript_text = " ".join(chunk["text"] for chunk in transcript_list)

    # print(transcript_text)

except TranscriptsDisabled:
    print("No caption is available for this video")


# Step 2: Test splitting
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([transcript_text])


# Step 3: Store the chunks into vector
client = chromadb.CloudClient(
  api_key=os.getenv("CHROMA_API_KEY"),
  tenant=os.getenv("CHROMA_TENANT"),
  database=os.getenv("CHROMA_DATABASE")
)

vector_store = Chroma(
    client=client,
    collection_name="my_collection6",
    embedding_function=embedding_model
)

# Add documents in batches

batch_size=100

for i in range(0, len(chunks),batch_size):
    batch=chunks[i:i+batch_size]
    # print(f"Adding chunks {i+1} to {i+len(batch)}....")
    vector_store.add_documents(batch)


print("All chunks successfully added!")

# Step 4: Construct  retriever
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})



# Step5 :Augmentation
llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)

prompt=PromptTemplate(
    template="""
    You are a helpfull assistent.
    answer only from the provide transcript context .
    If the context is insufficient simply say you don't know
 
    {context}
    Question:{question}
""",
input_variables=['context','question']
)

question="Is the topic of human reproduction discussed in this video ? if yes then what was discussed"
retrieve_docs=retriever.invoke(question)
# print(retrieve_docs)

context_text="\n\n".join(doc.page_content for doc in retrieve_docs)
final_prompt=prompt.invoke({"context":context_text,"question":question})

answer=llm.invoke(final_prompt)
print("ans",answer.content)







