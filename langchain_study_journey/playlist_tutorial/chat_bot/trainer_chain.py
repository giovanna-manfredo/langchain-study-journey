import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

template =  """Você é um personal trainer renomado e entende de todos os tipos de treinos para todos os tipos de \
fisicos. VOcê precisa responder à dúvidas do usuário sobre exercicios ou treinos. Seja amigável e detalhista. Apoie sempre \
seu aluno.
"""

prompt_template = ChatPromptTemplate([
    ("system", template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")

])

chain_trainer = prompt_template | llm | StrOutputParser()

