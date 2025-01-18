import config
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


async def conversation_chat():
    llm = ChatGoogleGenerativeAI(temperature=1, model="gemini-1.5-pro")

    conversation = [
    SystemMessage(content="Seja arrogante")
]

    while True:
        entrada = input("Entrada (digite q para parar): ")
        if entrada.lower() == "q":
            break
        
        conversation.append(HumanMessage(entrada))

        all_chunck = []
        async for chunk in llm.astream(conversation):
            all_chunck.append(chunk.content)
            print(chunk.content, end="", flush=True )

        resposta = "".join(all_chunck)

        conversation.append(AIMessage(resposta))

        

    print("fim da conversa")

asyncio.run(conversation_chat())
# conversation = [
#     SystemMessage(content="Seja arrogante")
# ]



# while True:
#     entrada = input("Entrada (digite q para parar): ")
#     if entrada.lower() == "q":
#         break
    
#     conversation.append(HumanMessage(entrada))

#     resultado = llm.invoke(conversation)
#     resposta = resultado.content

#     conversation.append(AIMessage(resposta))

#     print("IA resposta: " + resposta)

# print("fim da conversa")