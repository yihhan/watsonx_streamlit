import streamlit as st

import requests
import yaml

import os
import re
import json
import requests
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Any

def invoke_endpoint(payload, smclient, endpoint_name):
    res = smclient.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json",
                CustomAttributes="accept_eula=true")
    return res["Body"].read().decode("utf8")

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
          "First answer yes or no before explaining. "\
          "Cite each reference using [Page Number] notation (every result has this number at the beginning). "\
          "Citation should be done at the end of each sentence. Only include information found in the results and "\
          "don't add any additional information. Make sure the answer is correct and don't output false content. "\
          "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
          "search results which has nothing to do with the question. Only answer what is asked. The "\
          "answer should be short and concise."

  prompt += f"\n\n\nQuery: {question}\n\nAnswer: "

  return prompt
# # Load environment vars
load_dotenv()
ibm_api_key = os.getenv("IBM_API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)

# Title for the app
st.title('🤖 Compliance Checker')

model_option = st.selectbox('Choose your model?', 
                            ('Llama2-70b-chat', 'llama-2-13b-chat', 'llama-2-7b-chat'))

st.write('You selected:', model_option)

uploaded_file = st.file_uploader("Upload a YAML file", type=["yaml", "yml", "txt"])

uploaded_policy_file = st.file_uploader("Upload a policy file", type=["pdf"])

if model_option == "Llama2-70b-chat":
    model_id = "meta-llama/llama-2-70b-chat"

if model_option == "llama-2-13b-chat":
    model_id = "meta-llama/llama-2-13b-chat"

if model_option == "llama-2-7b-chat":
    model_id = "meta-llama/llama-2-7b-chat"

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
        st.subheader("Results Summary:")
        yaml_content = yaml.safe_load(uploaded_file.read())
        # st.write(yaml_content)

    df_results = pd.DataFrame(columns=["filename", "policy_file", "section", "compliance", "reasons"])

    output_policy = ""
    output_filename = uploaded_file.name
    output_section = ""
    output_compliance = ""
    output_reasons = ""

    if yaml_content:
        for key in yaml_content["Resources"]:

            #  st.write(yaml_content["Resources"][key])
            output_section = yaml_content["Resources"][key]

            if uploaded_policy_file is not None:

                output_policy = uploaded_policy_file.name

                prompt = "Input: You are an AWS cloud expert. What does this cloudformation template do? Answer concisely."
                prompt += "\n"
                prompt += str(yaml_content["Resources"][key])
                prompt += "\n"
                prompt += "Output:"

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
                # Pass the prompt to the llm
                #response = llm(prompt)
            
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]
                # Write the output to the screen
                # st.write(output)

                #Q&A functionalities
                question = output + "Is this compliant?"
                topn_chunks = get_search_results(question, embeddings, chunks)
                prompt = build_prompt(question, topn_chunks)
                prompt += "\n"
                prompt += "Output:"

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
                # Pass the prompt to the llm
                #response = llm(prompt)
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]
                # Write the output to the screen
                
                if "yes" in output[:5].lower():
                    output_compliance = "yes"
                if "no" in output[:5].lower():
                    output_compliance = "No"

                output_reasons = output
                new_row = {"filename":output_filename, "policy_file":output_policy, "section":output_section, "compliance":output_compliance, "reasons":output_reasons}
                df_results = df_results.append(new_row, ignore_index=True)

            else:
                prompt = "Input: You are a cyber security expect. Is the cloud formation template below CIS-compliant?  First answer yes or no before explaining." 
                prompt += "\n"
                prompt += str(yaml_content["Resources"][key])
                prompt += "\n"
                prompt += "Output:"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + ibm_api_key
                }

                payload = {
                    "model_id": model_id,
                    "inputs": [prompt],
                    "parameters": {"decoding_method": "greedy",  "max_new_tokens": 200,  "min_new_tokens": 0, "repetition_penalty": 1}
                }
                # Pass the prompt to the llm
                #response = llm(prompt)
                response = requests.request("POST", ibm_cloud_url, json=payload, headers=headers)
                response_json = response.json()
                output = response_json.get("results")[0]["generated_text"]
                # Write the output to the screen
                # st.write(output)
                output_reasons = output
                st.write(output)
                if "yes" in output[:5].lower():
                    output_compliance = "yes"
                if "no" in output[:5].lower():
                    output_compliance = "No"

                new_row = {"filename":output_filename, "policy_file":output_policy, "section":output_section, "compliance":output_compliance, "reasons":output_reasons}
                df_results = df_results.append(new_row, ignore_index=True)

    st.dataframe(df_results, use_container_width=True)
