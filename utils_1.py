from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.chains import RetrievalQA
from typing import Tuple, Dict
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import io
import base64
import requests
from PIL import Image
import streamlit as st

load_dotenv()

tavily_api_key = st.secrets["TAVILY_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings()




def encode_and_query_api(image, api_key):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe this image as good and authentic as possible , include all it details possible as it will be fed to and auto claim system to insurence companies"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']['content']





urls = [
    "https://en.wikipedia.org/wiki/Vehicle_insurance_in_the_United_States",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)

class Config():
    """
    Contains the configuration of the LLM.
    """
    model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(temperature=0, model=model)
cfg = Config()
qa = RetrievalQA.from_chain_type(
    llm=cfg.llm,
    chain_type="stuff",
    retriever = vectorstore.as_retriever()
)

def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the open ai functions agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    return agent_kwargs, memory

def setup_agent() -> AgentExecutor:
    """
    Sets up the tools for a function based chain.
    We have here the following tools:

    """
    cfg = Config()
    tools = [
        Tool(
            name="knowledge search",
            func=qa.run,
            description="useful for when you need more advanced search option to answer questions about insurence. "
        ),
        Tool(
        name='web search',
        func=TavilySearchResults(api_key=tavily_api_key).run,
        description=(
            '''use this tool when you can't find the content in the knowledge base and you need more advenced search functionalities  '''
        ))
        
    ]
    agent_kwargs, memory = setup_memory()

    return initialize_agent(
        tools,
        cfg.llm,
        verbose=False,
        agent_kwargs=agent_kwargs,
        memory=memory,
        handle_parsing_errors=True
    )

