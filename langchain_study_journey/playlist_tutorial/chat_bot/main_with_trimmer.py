from langchain_study_journey.playlist_tutorial.chat_bot.faq_chain import chain_faq
from langchain_study_journey.playlist_tutorial.chat_bot.memory_chat import get_session_history
from langchain_study_journey.playlist_tutorial.chat_bot.trainer_chain import chain_trainer
from langchain_study_journey.playlist_tutorial.chat_bot.roteamento_chain import chain_route
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableWithMessageHistory

from operator import itemgetter

from langchain_core.messages import trim_messages

def choice_route(choice_dict:dict) -> RunnableLambda:
    if choice_dict["pydantic_choice"].choice == True:
        return RunnableLambda(lambda x: {"input": x["input"], "history": x["history"]}) | chain_faq
    else:
        return RunnableLambda(lambda x: {"input": x["input"], "history": x["history"]}) | chain_trainer

trimmer = trim_messages(strategy="last", max_tokens=2, token_counter=len)



main_chain = RunnableParallel({
    "input": itemgetter("input"),
    "history": itemgetter("history"),
    "pydantic_choice": chain_route
}) | choice_route

trimmer_chain = RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | main_chain

runnable_with_history = RunnableWithMessageHistory(
    trimmer_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

result = runnable_with_history.invoke(
    {"input": "O que devo treinar sendo que tenho 1.66 e 66kg?"},
    config={"configurable": {"session_id": "2"}},
)

print(result)