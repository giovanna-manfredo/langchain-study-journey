from langchain_community.chat_message_histories import SQLChatMessageHistory

from sqlalchemy.ext.asyncio import create_async_engine

async_engine = create_async_engine("sqlite+aiosqlite:///memoryaio.db")

def get_session_history(session_id:id) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection=async_engine)