from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http.models import Distance, VectorParams

from pathlib import Path

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 

llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

client = QdrantClient(
    url="https://ece21834-31bb-4880-9479-ba1d734e2185.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="K3qrHZ_0LElWb9oYxngRKzgNGH1YH5ukv9tjDNBjJc7bF9VAyla3jQ",
)


client.create_collection(
    collection_name="test",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)


vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings
)

loader = PyPDFLoader(Path(__file__).parent.parent / "docs"/ "layout-parser-paper.pdf")
docs = loader.load()

text_splitter = SemanticChunker(embeddings)

# Aplicando o split
texts = text_splitter.split_documents(docs)

vector_store.add_documents(documents=texts)

query = "Explique os principais tópicos abordados no documento."
query_embedding = embeddings.embed_query(query)

results = vector_store.similarity_search_by_vector(
    query_embedding,
    k=5  # Retornar os 5 documentos mais similares
)

retrieved_text = "\n\n".join([doc.page_content for doc in results])

prompt_template_str = "com base em {retrived_text}, responda {query}"

prompt = ChatPromptTemplate([prompt_template_str],
                            partial_variables={"query": query})

chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {"retrived_text": retrieved_text}
)
print(result)