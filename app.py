# Import environment loading library
#from dotenv import load_dotenv
# Import IBMGen Library 
#from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
#from langchain.llms.base import LLM
# Import lang Chain Interface object
#from langChainInterface import LangChainInterface
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st
import requests


# # Load environment vars
# load_dotenv()

# # Define credentials 
# api_key = os.getenv("API_KEY", None)
# ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
# project_id = os.getenv("PROJECT_ID", None)
# if api_key is None or ibm_cloud_url is None or project_id is None:
#     print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
# else:
#     creds = {
#         "url": ibm_cloud_url,
#         "apikey": api_key 
#     }

# Define generation parameters 
# params = {
#     GenParams.DECODING_METHOD: "sample",
#     GenParams.MIN_NEW_TOKENS: 30,
#     GenParams.MAX_NEW_TOKENS: 300,
#     GenParams.TEMPERATURE: 0.2,
#     # GenParams.TOP_K: 100,
#     # GenParams.TOP_P: 1,
#     GenParams.REPETITION_PENALTY: 1
# }

# define LangChainInterface model
#llm = LangChainInterface(model='google/flan-ul2', credentials=creds, params=params, project_id=project_id)

# Title for the app
st.title('🤖 Compliance Checker')
# Prompt box 
prompt = st.text_area('Enter your prompt here')
# If a user hits enter
if prompt: 
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer pak-EZDdAZYO7HAXbDu1O-CB5BiBz-k-vGnWdtJj-yfJtdg"
    }

    payload = {
        "model_id": "meta-llama/llama-2-70b-chat",
        "inputs": [prompt],
        "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
    }
    # Pass the prompt to the llm
    #response = llm(prompt)
    response = requests.request("POST", "https://bam-api.res.ibm.com/v1/generate", json=payload, headers=headers)
    response_json = response.json()
    output = response_json.get("results")[0]["generated_text"]
    # Write the output to the screen
    st.write(output)