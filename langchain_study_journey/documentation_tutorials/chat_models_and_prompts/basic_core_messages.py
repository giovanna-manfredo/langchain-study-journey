import config

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import HumanMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

messages = [
    SystemMessage("Give me 5 good songs of the mentioned genre"),
    HumanMessage("Rock"),
]

response = model.invoke(messages)  # This is a AIMessage (look like a assistent)

response2 = model.invoke(
    [
        {
            "role": "system",
            "content": "You are a travel assistant. Suggest travel destinations based on the user's interests.",
        },
        {
            "role": "user",
            "content": "I like places with beautiful beaches and warm weather.",
        },
        {
            "role": "assistant",
            "content": "Based on your interest in beaches and warm weather, I would recommend destinations like the Maldives, Hawaii, or Rio de Janeiro.",
        },
    ]
)

print(response.content)
print(response2.content)
