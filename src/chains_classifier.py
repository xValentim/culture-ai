from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model_name_groq = "llama-3.2-90b-text-preview"
model_name_openai = "gpt-4o-2024-08-06"
"""
venus: https://lumalabs.ai/embed/18947f38-0421-47fa-bb20-51c658a144b7?mode=sparkles&background=%23ffffff&color=%23000000&cinematicVideo=undefined&showMenu=false

amnesia/Flavio cerqueira: https://lumalabs.ai/embed/e6648ddb-3bc2-4f8e-84bb-12d42b4731bc?mode=sparkles&background=%23ffffff&color=%23000000&showTitle=true&loadBg=true&logoPosition=bottom-left&infoPosition=bottom-right&cinematicVideo=undefined&showMenu=false

Gi: https://splat-vis-demo.up.railway.app/?url=https://huggingface.co/xValentim/splat-masp/resolve/main/hacka.splat?download=true

""" 

mapping = {
    "venus": "https://lumalabs.ai/embed/18947f38-0421-47fa-bb20-51c658a144b7?mode=sparkles&background=%23ffffff&color=%23000000&cinematicVideo=undefined&showMenu=false",
    "amnesia": "https://lumalabs.ai/embed/e6648ddb-3bc2-4f8e-84bb-12d42b4731bc?mode=sparkles&background=%23ffffff&color=%23000000&showTitle=true&loadBg=true&logoPosition=bottom-left&infoPosition=bottom-right&cinematicVideo=undefined&showMenu=false",
    "gi": "https://splat-vis-demo.up.railway.app/?url=https://huggingface.co/xValentim/splat-masp/resolve/main/hacka.splat?download=true",
    "none": None
}



llm_openai = ChatOpenAI(
    model=model_name_openai, # 100% json output
    temperature=0,
)

llm_groq = ChatGroq(
    model=model_name_groq, 
    temperature=0,
)

system_prompt = """

Você é um assistente de inteligência artificial que auxilia na classificação de quais obras o usuário quer ver um objeto 3D. O usuário fornece descrição e você precisa classificar qual objeto 3D ele deseja ver (ou se ele não deseja ver nenhum objeto). Os objetos que possuimos cenas 3D e que devem ser classificados são: venus, amnesia, gi e none.

- venus: Se refere a Venus Victrix ou Vênus Vitoriosa de Pierre-Auguste Renoir.
- amnesia: Se refere a Amnésia de Flávio Cerqueira.
- gi: Se refere a giovanna moeller, uma desenvolvedora de software.
- none: Se o usuário não deseja ver nenhum objeto.

"""

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), 
            ("human", "query do usuário: \n\n {query}")
        ]
)

class GetSchema(BaseModel):
    """Classifica qual objeto 3D o usuário deseja ver (ou se ele não deseja ver nenhum objeto)"""
    
    resultado: str = Field(description="Resultado da classificação do objeto 3D que o usuário deseja ver (ou se ele não deseja ver nenhum objeto)", examples=['venus', 'amnesia', 'gi', 'none'])
    

llm_openai_with_tools_extraction = llm_openai.bind_tools([GetSchema]) #, strict=True)
llm_groq_with_tools_extraction = llm_groq.with_structured_output(GetSchema)

chain_openai_structured_extraction = prompt | llm_openai_with_tools_extraction
chain_groq_structured_extraction = prompt | llm_groq_with_tools_extraction

def classifier_3d_object(query: str, model: str='groq'):
    if model == 'groq':
        try:
            response_groq = chain_groq_structured_extraction.invoke({"query": query})
            obra = response_groq.resultado
            return mapping[obra]
        except:
            return None
    else:
        try:
            response_openai = chain_openai_structured_extraction.invoke({"query": query})
            obra = response_openai.tool_calls[0]['args']['resultado']
            return mapping[obra]
        except:
            return None