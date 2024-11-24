from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from operator import itemgetter 
from langchain_openai import OpenAIEmbeddings

"""

Modelos disponíveis:

- mixtral-8x7b-32768
- llama-guard-3-8b -> Somente para guardrails
- llama-3.2-90b-text-preview
- llama3-70b-8192
- llama-3.1-70b-versatile

"""

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

vdb_masp = FAISS.load_local("vectorstore/masp", 
                            embeddings, 
                            allow_dangerous_deserialization=True)

vdb_mexico = FAISS.load_local("vectorstore/mexico", 
                            embeddings, 
                            allow_dangerous_deserialization=True)


retriever_masp = vdb_masp.as_retriever(search_kwargs={"k": 2})
retriever_mexico = vdb_mexico.as_retriever(search_kwargs={"k": 2})

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)


system_prompt = """
Você é um assistente de IA que vai tirar dúvidas de cultura geral do usuário. 

Aqui está o histórico da conversa:

[Inicio do histórico]
{chat_history}
[Final do histórico]

Além disso, aqui está um contexto extra sobre o masp (que você pode ou não usar):

[Contexto extra]
{context_masp}
[Final do contexto extra]

Além disso, aqui está um contexto extra sobre o México (que você pode ou não usar):

[Contexto extra]
{context_mexico}
[Final do contexto extra]

--------------------------------------------
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

chain = (
    {
        "chat_history": itemgetter("chat_history"),
        "input": itemgetter("input"),
        "context_masp": itemgetter("input") | retriever_masp,
        "context_mexico": itemgetter("input") | retriever_mexico,
    }
    | prompt 
    | llm 
    | StrOutputParser()
)
