import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke(
    {"language": "Italian", "text": "How are you?!"}
)  # This is a ChatPromptValue

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

response = llm.invoke(prompt)

print(response.content)
