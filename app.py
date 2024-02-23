from openai import OpenAI
import streamlit as st
from PIL import Image
from utils_1 import *
import base64
import requests
import io
from langchain.prompts import PromptTemplate


agent_executor = setup_agent()

template="""You are insurAI an expert in assitant used by top insurence companies your main role is to help the clients in their accident .
you can answer anyhting else and you answers should always be helpful and provide assitant for the users
start by doing checking the images the user provided 
remember to give helpful advices then keep ask follow up questions to the users until you have a clear understanding of the problem they are having 
you can also estimate how much their insurence will give them back based on the damage of the car and the situation 
You have access to the following tools:
\n\nknowledge search: useful for when you need information about questions about insurance. 
\nweb search: use this tool when you can't find the content in the knowledge base and you need more advanced search functionalities
\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [knowledge search, web search]\n
Action Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\n
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question\n\nBegin!\n\n
Question: {input}\nThought:{agent_scratchpad}"""
promptup = PromptTemplate(
    input_variables=['agent_scratchpad', 'input'],
    template = template
)

agent_executor.agent.llm_chain.prompt = promptup

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
api_key = st.secrets["OPENAI_API_KEY"]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    uploaded_file = None
    agent_executor.memory.clear()

# App title
st.set_page_config(page_title="InsurAI is Here 🦺")
logo_image_path = "logo.png"
logo_image = Image.open(logo_image_path)
resized_logo_image = logo_image.resize((250, 250))  # Adjust the width and height as needed
st.image(resized_logo_image, use_column_width=True)
# Replicate Credentials
with st.sidebar:
    st.title('Meet InsurAI 🤖🦺')
    st.markdown(""" 
Our app transforms the auto insurance journey, simplifying claims and post-accident guidance for enhanced customer satisfaction. As technology reshapes the industry, the app addresses evolving challenges, offering a streamlined solution to modernize the auto insurance experience.
""")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


avatar_img="avatar.png"

# Set OpenAI key
client = OpenAI(api_key=api_key)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] == "user" :
      with st.chat_message(message["role"]):
          st.write(message["content"])
    else : 
      with st.chat_message(message["role"],avatar = avatar_img ):
          st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
  result = ' '
  if uploaded_file is not None :
    image = Image.open(uploaded_file)
    result = encode_and_query_api(image, api_key)
  with st.chat_message("assistant", avatar="logo.png"):
    st.markdown(result)
    with st.spinner("Thinking..."):
      response = agent_executor.invoke(result + " " +  st.session_state.messages[-1]["content"])
      placeholder = st.empty()
      full_response = ''
      for item in response['output']:
        full_response += item
      placeholder.markdown(full_response)
    placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response + result}
    st.session_state.messages.append(message)
    uploaded_file = None
