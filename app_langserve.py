#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from dotenv import load_dotenv
from src.chains_culture import *

load_dotenv()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# system = """

# Você é um assistente muito útil que irá me ajudar a estudar para o ENEM.

# """

# llm = ChatGroq(model="llama-3.2-90b-text-preview", temperature=0)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system),
#     ("human", "input: {input}")
# ])

# chain = prompt | llm | StrOutputParser()

add_routes(
    app,
    chain,
    path="/culture",
)

