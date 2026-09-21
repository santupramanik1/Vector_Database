import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key=os.getenv("OPENROUTER_API_KEY")

connection=OpenAIEmbeddings(
    model="liquid/lfm-2.5-embedding-350m:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False
)
