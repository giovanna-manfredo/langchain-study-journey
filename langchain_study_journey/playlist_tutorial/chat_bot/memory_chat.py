from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: int) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_bot.db")

