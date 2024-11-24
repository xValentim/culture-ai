from typing import Union, Literal
from pydantic import BaseModel
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from src.chains_culture import chain
from src.chains_classifier import *
from fastapi.responses import StreamingResponse
import json

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
    
    obra_3d = classifier_3d_object(last_message, model='groq')
    additional_kwargs = {'URL': obra_3d} if obra_3d else {}

    print(additional_kwargs, "ADITIONAL")
    
    chat_history = []
    for message in converted_messages[-6:-1]:
        if isinstance(message, AIMessage):
            chat_history.append(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            chat_history.append(f"Human: {message.content}")
    
    print(chat_history)

    async def generate():
        # Send URL first if it exists
        if obra_3d:
            yield f"data: {json.dumps({'type': 'url', 'content': obra_3d})}\n\n"
        
        async for chunk in chain.astream({"input": last_message, "chat_history": chat_history}):
            yield f"data: {json.dumps({'type': 'message', 'content': chunk})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )