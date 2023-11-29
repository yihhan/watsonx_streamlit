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
import yaml

import os
import re
import requests

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Any

# Load the model from TF Hub
class MiniLML6V2EmbeddingFunction():
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_function = MiniLML6V2EmbeddingFunction()

def pdf_to_text(path: str,
                start_page: int = 1,
                end_page: Optional[int | None] = None) -> list[str]:
  """
  Converts PDF to plain text.

  Args:
      path (str): Path to the PDF file.
      start_page (int): Page to start getting text from.
      end_page (int): Last page to get text from.
  """
  loader = PyPDFLoader(path)
  pages = loader.load()
  total_pages = len(pages)

  if end_page is None:
      end_page = len(pages)

  text_list = []
  for i in range(start_page-1, end_page):
      text = pages[i].page_content
      text = text.replace('\n', ' ')
      text = re.sub(r'\s+', ' ', text)
      text_list.append(text)

  return text_list

def text_to_chunks(texts: list[str],
                   word_length: int = 150,
                   start_page: int = 1) -> list[list[str]]:
  """
  Splits the text into equally distributed chunks.

  Args:
      texts (str): List of texts to be converted into chunks.
      word_length (int): Maximum number of words in each chunk.
      start_page (int): Starting page number for the chunks.
  """
  text_toks = [t.split(' ') for t in texts]
  chunks = []

  for idx, words in enumerate(text_toks):
      for i in range(0, len(words), word_length):
          chunk = words[i:i+word_length]
          if (i+word_length) > len(words) and (len(chunk) < word_length) and (
              len(text_toks) != (idx+1)):
              text_toks[idx+1] = chunk + text_toks[idx+1]
              continue
          chunk = ' '.join(chunk).strip()
          chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
          chunks.append(chunk)

  return chunks

def get_text_embedding(texts: list[list[str]],
                       batch: int = 1000) -> list[Any]:
  """
  Get the embeddings from the text.

  Args:
      texts (list(str)): List of chucks of text.
      batch (int): Batch size.
  """
  embeddings = []
  for i in range(0, len(texts), batch):
      text_batch = texts[i:(i+batch)]
      # Embeddings model
      emb_batch = emb_function(text_batch)
      embeddings.append(emb_batch)
  embeddings = np.vstack(embeddings)
  return embeddings

def get_search_results(question, embeddings, chunks):
  """
  Get best search results
  """
  emb_question = emb_function([question])
  nn = NearestNeighbors(n_neighbors=3)
  nn.fit(embeddings)
  neighbors = nn.kneighbors(emb_question, return_distance=False)
  topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]

  return topn_chunks

def build_prompt(question, topn_chunks_for_prompts):

  '''
  build prompt for general Q&A
  '''

  prompt = ""
  prompt += 'Search results:\n'

  for c in topn_chunks_for_prompts:
      prompt += c + '\n\n'

  prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
          "Cite each reference using [Page Number] notation (every result has this number at the beginning). "\
          "Citation should be done at the end of each sentence. Only include information found in the results and "\
          "don't add any additional information. Make sure the answer is correct and don't output false content. "\
          "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
          "search results which has nothing to do with the question. Only answer what is asked. The "\
          "answer should be short and concise."

  prompt += f"\n\n\nQuery: {question}\n\nAnswer: "

  return prompt
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

model_option = st.selectbox('Choose your model?', 
                            ('Llama2-70b-chat', 'llama-2-13b-chat', 'llama-2-7b-chat'))
st.write('You selected:', model_option)

uploaded_file = st.file_uploader("Upload a YAML file", type=["yaml", "yml", "txt"])

uploaded_policy_file = st.file_uploader("Upload a policy file", type=["pdf"])

if uploaded_policy_file is not None:
    with open("temp_save", 'wb') as f: 
        f.write(uploaded_policy_file.getvalue())

    # Get embedding
    text_list = pdf_to_text("temp_save")
    chunks = text_to_chunks(text_list)
    embeddings = get_text_embedding(chunks)

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.write("File Details:")
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size} bytes"}
    st.write(file_details)

    # If the uploaded file is a YAML file, read and display its content
    if uploaded_file.type in ["application/x-yaml", "text/yaml", "text/plain"]:
        st.subheader("YAML Content:")
        yaml_content = yaml.safe_load(uploaded_file.read())
        # st.write(yaml_content)

    if yaml_content:
        for key in yaml_content["Resources"]:
            st.write(yaml_content["Resources"][key])

            if uploaded_policy_file is not None:
                prompt = "Input: You are an AWS cloud expert. What does this cloudformation template do? Answer concisely." 
                prompt += "\n"
                prompt += str(yaml_content["Resources"][key])
                prompt += "\n"
                prompt += "Output:"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer pak-EZDdAZYO7HAXbDu1O-CB5BiBz-k-vGnWdtJj-yfJtdg"
                }

                if model_option == "Llama2-70b-chat":
                    model_id = "meta-llama/llama-2-70b-chat"

                if model_option == "llama-2-13b-chat":
                    model_id = "meta-llama/llama-2-13b-chat"

                if model_option == "llama-2-7b-chat":
                    model_id = "meta-llama/llama-2-7b-chat"

                payload = {
                    "model_id": model_id,
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

                #Q&A functionalities
                question = output + "Is this compliant?"
                topn_chunks = get_search_results(question, embeddings, chunks)
                prompt = build_prompt(question, topn_chunks)
                prompt += "\n"
                prompt += "Output:"

                # prompt = "Input: You are a cyber security expect. Is the cloud formation template below CIS-compliant?" 
                # prompt += "\n"
                # prompt += output
                # prompt += "\n"
                # prompt += str(yaml_content["Resources"][key])
                # prompt += "\n"
                # prompt += "Output:"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer pak-EZDdAZYO7HAXbDu1O-CB5BiBz-k-vGnWdtJj-yfJtdg"
                }

                if model_option == "Llama2-70b-chat":
                    model_id = "meta-llama/llama-2-70b-chat"

                if model_option == "llama-2-13b-chat":
                    model_id = "meta-llama/llama-2-13b-chat"

                if model_option == "llama-2-7b-chat":
                    model_id = "meta-llama/llama-2-7b-chat"

                payload = {
                    "model_id": model_id,
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

            else:
                prompt = "Input: You are a cyber security expect. Is the cloud formation template below CIS-compliant?" 
                prompt += "\n"
                prompt += str(yaml_content["Resources"][key])
                prompt += "\n"
                prompt += "Output:"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer pak-EZDdAZYO7HAXbDu1O-CB5BiBz-k-vGnWdtJj-yfJtdg"
                }

                if model_option == "Llama2-70b-chat":
                    model_id = "meta-llama/llama-2-70b-chat"

                if model_option == "llama-2-13b-chat":
                    model_id = "meta-llama/llama-2-13b-chat"

                if model_option == "llama-2-7b-chat":
                    model_id = "meta-llama/llama-2-7b-chat"

                payload = {
                    "model_id": model_id,
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

# # Prompt box 
# prompt = st.text_area('Enter your prompt here')
# # If a user hits enter
# if prompt: 
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer pak-EZDdAZYO7HAXbDu1O-CB5BiBz-k-vGnWdtJj-yfJtdg"
#     }

#     if model_option == "Llama2-70b-chat":
#         model_id = "meta-llama/llama-2-70b-chat"

#     if model_option == "llama-2-13b-chat":
#         model_id = "meta-llama/llama-2-13b-chat"

#     if model_option == "llama-2-7b-chat":
#         model_id = "meta-llama/llama-2-7b-chat"

#     payload = {
#         "model_id": model_id,
#         "inputs": [prompt],
#         "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
#     }
#     # Pass the prompt to the llm
#     #response = llm(prompt)
#     response = requests.request("POST", "https://bam-api.res.ibm.com/v1/generate", json=payload, headers=headers)
#     response_json = response.json()
#     output = response_json.get("results")[0]["generated_text"]
#     # Write the output to the screen
#     st.write(output)
