import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# Set Wikipedia language
wikipedia.set_lang("en")

# Add a proper User-Agent to the underlying requests session 
#Note: without telling user agent wikipedia discard the request
wikipedia.wikipedia.USER_AGENT = (
    "VectorDatabaseLearning/1.0 (student project)"
)

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "Geographical history of India and Pakistan"

docs = retriever.invoke(query)

print(docs)

for i, doc in enumerate(docs):
    print(f"\n---- Result {i + 1} ----")
    print(f"Content:\n{doc.page_content}...")