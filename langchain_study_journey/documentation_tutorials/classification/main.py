import config

from pydantic import BaseModel, Field

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive"
    )
    language: str = Field(description="The language the text is written in")


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0).with_structured_output(Classification)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(response)