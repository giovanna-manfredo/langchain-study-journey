import config

from typing import Optional

from pydantic import BaseModel, Field

from langchain.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

structured_llm = llm.with_structured_output(schema=Person)

text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)

print(response)