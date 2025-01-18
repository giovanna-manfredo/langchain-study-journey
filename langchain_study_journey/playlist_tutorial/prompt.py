# prompt template só formata a string
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = PromptTemplate.from_template("gere um poema sobre {topico}. Escreva em {lingua}")
retorno = prompt.invoke({"topico": "borbolrtas", "lingua": "inglês"}) 

print(retorno)

# formata e ja sabe que é do tipo humano
from langchain_core.prompts import ChatPromptTemplate
#HumanMessagePromptTemplate.from_template - forma explicita ou tuplas(mais facil): ("role": "value")
prompt = ChatPromptTemplate(["gere um poema sobre {topico}. Escreva em {lingua}"])
retorno = prompt.invoke({"topico": "borbolrtas", "lingua": "inglês"}) 

print(retorno)


prompt = ChatPromptTemplate([
    ("system", "você é um poeta"),
    ("user", "me de um podema de {topico} em {lingua}")
])

retorno = prompt.invoke({"topico": "borbolrtas", "lingua": "inglês"}) 


prompt = ChatPromptTemplate([
    ("system", "você é um poeta"),
    MessagesPlaceholder("msg")
])

retorno = prompt.invoke({"msg": [HumanMessage(content= "me faça um poema de borboletas")]}) 

print(retorno)