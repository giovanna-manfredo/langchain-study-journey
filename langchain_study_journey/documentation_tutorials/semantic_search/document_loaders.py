import config

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Loud document
file_path = Path(__file__).parent / "files" / "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

#splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splitters = text_splitter.split_documents(docs)

print(len(all_splitters))



#Embeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_1 = embeddings.embed_query(all_splitters[0].page_content)
vector_2 = embeddings.embed_query(all_splitters[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length of dimensions: {len(vector_1)}\n")
print(vector_1[:10])

#vector store in memory

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splitters)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(results[0])

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])