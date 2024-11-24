from typing import Union, Literal
from pydantic import BaseModel
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from src.chains_culture import chain
from fastapi.responses import StreamingResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageBase(BaseModel):
    type: Literal["human", "ai"]
    content: str
    additional_kwargs: dict = {}
    example: bool = False

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
async def chat(
    messages: list[MessageBase]
):
    # Convert the messages to LangChain message types
    converted_messages = []
    for msg in messages:
        if msg.type == "human":
            converted_messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            converted_messages.append(AIMessage(content=msg.content))

    last_message = converted_messages[-1].content
    
    chat_history = []
    for message in converted_messages[-6:-1]:
        if isinstance(message, AIMessage):
            chat_history.append(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            chat_history.append(f"Human: {message.content}")
    
    print(chat_history)

    async def generate():
        async for chunk in chain.astream({"input": last_message, "chat_history": chat_history}):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )