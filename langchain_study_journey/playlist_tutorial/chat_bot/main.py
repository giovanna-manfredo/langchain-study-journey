from langchain_study_journey.playlist_tutorial.chat_bot.faq_chain import chain_faq
from langchain_study_journey.playlist_tutorial.chat_bot.trainer_chain import chain_trainer
from langchain_study_journey.playlist_tutorial.chat_bot.roteamento_chain import chain_route

from langchain_study_journey.playlist_tutorial.chat_bot.memory_chat import get_session_history
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory, RunnableParallel

from operator import itemgetter

def choice_route(route_dict:dict) -> RunnableLambda:
    if route_dict['response_pydantic'].choice == True:
        return RunnableLambda(lambda x: {"input": x["input"],"history": x["history"]}) | chain_faq
    else:
        return RunnableLambda(lambda x: {"input": x["input"],"history": x["history"]}) | chain_trainer
    
main_chain = RunnableParallel({
    "input": itemgetter("input"),
    "history": itemgetter("history"),
    "response_pydantic": chain_route
}) | RunnableLambda(choice_route)
    
runnable_with_message_history = RunnableWithMessageHistory(
    main_chain,
    get_session_history,
    history_messages_key="history",
    input_messages_key="input"
)

result = runnable_with_message_history.invoke(
    {"input": "Quais os planos da academia?"},
    config={"configurable": {"session_id": "1"}},
)

print(result)
