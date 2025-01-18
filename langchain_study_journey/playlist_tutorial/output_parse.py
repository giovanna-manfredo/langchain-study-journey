import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

class Steps(BaseModel):
    id:int = Field(description="Order of Number to my step")
    step_description:str = Field(description="Description my step")

class Task(BaseModel):
    task_name: str = Field(description="name of my task")
    steps: list[Steps] = Field(description="list of steps to task")

parser = JsonOutputParser(pydantic_object=Task)

prompt = PromptTemplate(
    template="Break my task in 8 steps.\n{format_instructions}\n{query}\n ",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

response = chain.invoke({"query": "Pay my count"})

print(response)

# # Define your desired data structure.
# class Joke(BaseModel):
#     setup: str = Field(description="question to set up a joke")
#     punchline: str = Field(description="answer to resolve the joke")

 
# # Set up a parser + inject instructions into the prompt template.
# parser = JsonOutputParser(pydantic_object=Joke)

# prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# # And a query intended to prompt a language model to populate the data structure.
# prompt_and_model = prompt | llm |parser
# output = prompt_and_model.invoke({"query": "Tell me a joke."})
# print(output)