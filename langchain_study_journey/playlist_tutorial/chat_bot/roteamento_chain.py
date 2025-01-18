import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

class Route(BaseModel):
    choice: bool = Field(description="Defina True se a pergunta do usuário for referente à dúvidas gerais de um FAQ. \
Defina False se for uma solicitação de ajuda para montar um treino ou pergunta específica sobre um exercicio ou treino.") 

parser = PydanticOutputParser(pydantic_object=Route)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

template = """Você é um especialista em classificação. Você receberá perguntas do usuário e precisará classificar, \
de forma booleana, se o usuário está perguntando sobre dúvidas gerais sobre a academia e planos ou se ele precisa \
de ajuda com um treino ou exercício.
\n{format_instructions}\n
Pergunta Usuário: {input}"
"""

prompt_template = ChatPromptTemplate([template], partial_variables={"format_instructions": parser.get_format_instructions()})

chain_route = prompt_template | llm | parser