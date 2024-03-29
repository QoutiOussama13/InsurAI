from openai import OpenAI
import streamlit as st
from PIL import Image
from utils_1 import *
import base64
import requests
import io
from langchain.prompts import PromptTemplate


agent_executor = setup_agent()

template="""InsurAI, you're here to assist clients through car accidents and insurance claims. A user has provided images of their recent accident. Begin by thoroughly examining the images to assess the extent of the damage. Offer guidance on steps to take immediately after an accident, ensuring safety and compliance with legal requirements.
Additionally, estimate the potential insurance coverage based on the visible damage and the user's situation.
Provide examples of policies relevant to the user's insurance coverage, detailing the types of damages typically covered and any deductible amounts. Offer reimbursement estimates for common expenses such as towing ($150), rental car fees ($30 per day), and medical bills (up to $5,000), tailored to the user's policy. Include pertinent details such as claim submission procedures, contact information for claims representatives, 
and approximate processing times (typically 7-10 business days). 
Remember, your responses should always be helpful and aimed at empowering the user to navigate through this challenging situation effectively.
Include as detailed answer as possible 
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
    with st.spinner("Thinking..."):
        st.write(result)
        content = result + str(st.session_state.messages[-1]["content"])
        response = agent_executor.invoke(content)
    placeholder = st.empty()
    full_response = ''
    for item in response['output']:
        full_response += item
        placeholder.markdown(full_response)
    placeholder.markdown(full_response)
  message = {"role": "assistant", "content": full_response + result}
  st.session_state.messages.append(message)
  uploaded_file = None
