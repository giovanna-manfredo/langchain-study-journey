from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

documents = [
    "Olá!",
    "Quantos anos você tem?",
    "Qual seu nome?",
    "Meu amigo se chama flávio",
    "Oi!"
]

embeddings = embeddings_model.embed_documents(documents)

print(len(embeddings))        
print(len(embeddings[0]))     # Cada embedding costuma ter 3072 dimensões, dependendo do modelo


embedded_query = embeddings_model.embed_query("Qual é o nome do seu amigo?")

print(len(embedded_query))  # Tamanho do vetor da query (ex. 3072)

print(embeddings)  # Tamanho do vetor da query (ex. 3072)
