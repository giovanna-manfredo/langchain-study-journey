import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough


# Inicializando o modelo LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

# Criando os templates de prompt
prompt_template_negativo = ChatPromptTemplate([
     "Você é um critico de cinema, fale mal do filme {movie}"
])

prompt_template_positivo = ChatPromptTemplate([
    "Você é um critico de cinema, fale bem do filme {movie}"
])

# Criando as cadeias para os prompts positivos e negativos
chain_positiva = prompt_template_positivo | llm | StrOutputParser()
chain_negativa = prompt_template_negativo | llm | StrOutputParser()

def analise(dict:dict):
    prompt_analise = ChatPromptTemplate([
     "de uma nota final ao filme, com base nos pontos fortes: {dpositivo} e negativo: {dnegativo}"
    ])
    chain = prompt_analise | llm | StrOutputParser()

    result = chain.invoke({"dpositivo": dict["positivo"], "dnegativo": dict["negativo"]})

    return result

# Utilizando o RunnableParallel para executar ambas as cadeias simultaneamente
chain = RunnableParallel(positivo=chain_positiva, negativo=chain_negativa) | RunnableLambda(analise)

# Invocando as cadeias com o filme especificado
result = chain.invoke({"movie": "barbie"})

# Exibindo o 
# resultado
print(result)