from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

def get_session_history(session_id: int):
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///memory.db")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

prompt_template = ChatPromptTemplate([
    ("system", "Você é alguem simpatico "),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Quero conselhos sobre {input}"),
])

main_chain = prompt_template | llm | StrOutputParser()

trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)

trim_messages_chain = (RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | main_chain)

runnable_with_history = RunnableWithMessageHistory(
    main_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

result = runnable_with_history.invoke(
    {"input":"Qual foi minha 1 pergunta?"},
    config={"configurable":{"session_id":"1"}},
)

print(result)