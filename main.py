import json
import torch
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer, AutoModel


#### OPENAI API ####

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key


#### DATA ####

# Read data from jsonl file
file_path_texts = "data/statements_id_statement2paragraphs_sub.jsonl"
file_path_titles = "data/statements_id_title_sub.jsonl"
texts = []
titles = []
ids = []
with open(file_path_titles, "r") as file:
    for line in file:
        data = json.loads(line)
        titles.append(data["title"])
        ids.append(data["id"])

with open(file_path_texts, "r") as file:
    for line in file:
        data = json.loads(line)
        texts.append(data["statement2paragraphs"])

print(f"Loaded {len(titles)} titles.")



#### VECTORIZE ####

# # Function to encode text to a vector using OpenAI's embedding API
# def encode_text_to_vector(text):
#     response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
#     return response.data[0].embedding


# Load the tokenizer and model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to encode text to a vector using the chosen model
def encode_text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Vectorize documents
vectors = [encode_text_to_vector(title) for title in titles]

print(f"Embeddings created for {len(titles)} titles.")


#### FAISS INDEX ####

# Dimension of vectors
dim = len(vectors[0])

# Creating the FAISS index
index = faiss.IndexFlatL2(dim)
index.add(np.array(vectors).astype("float32"))  # FAISS expects float32
print(f"FAISS index contains {index.ntotal} vectors.")



#### RETRIEVE ####

# Define a function to retrieve documents
def retrieve_documents(query, index, documents, top_k=1):
    # Vectorize the query using the same approach as for the documents
    query_vector = encode_text_to_vector(query)

    # Search the index for the most similar document vectors
    D, I = index.search(np.array([query_vector]).astype("float32"), top_k)

    # Retrieve the corresponding documents
    retrieved_docs = [documents[i] for i in I[0]]
    return retrieved_docs


#### PROMPT ####

# Define a query
query_question = "What was the revenue for UP Fintech in the second quater of 2023?"
retrieved_title = retrieve_documents(query_question, index, titles)
print(retrieved_title)

# Find statement based on title
# Get the index of the retrieved title
retrieved_title_index = titles.index(retrieved_title[0])
retrieved_text = texts[retrieved_title_index]

# Define the prompt
system_prompt = [
    {
        "role": "system",
        "content": "Hello, I am a financial analyst. My expertise is answering questions about financial statements.",
    }
]
user_prompt = [
    {
        "role": "user",
        "content": "Based on the financial statement below, please answer the following question: "
        + query_question
        + "\n\n"
        + retrieved_text,
    }
]
conversation = system_prompt + user_prompt

# Call the OpenAI API
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=conversation,
)

# Extract and return the model's reply
reply = response.choices[0].message.content
print(reply)

# query_question = st.text_input("Enter your query about financial statements:")

# # Button to trigger analysis
# if st.button('Analyze'):
#     if query_question:
#         retrieved_title = retrieve_documents(query_question, index, titles)
        
#         # Find statement based on title
#         # Get the index of the retrieved title
#         retrieved_title_index = titles.index(retrieved_title[0])
#         retrieved_text = texts[retrieved_title_index]

#         # Define the prompt
#         system_prompt = [
#             {
#                 "role": "system",
#                 "content": "Hello, I am a financial analyst. My expertise is answering questions about financial statements.",
#             }
#         ]
#         user_prompt = [
#             {
#                 "role": "user",
#                 "content": "Based on the financial statement below, please answer the following question: "
#                 + query_question
#                 + "\n\n"
#                 + retrieved_text,
#             }
#         ]
#         conversation = system_prompt + user_prompt
#         print(conversation)

#         # Call the OpenAI API
#         response = openai.chat.completions.create(
#             model="gpt-4-1106-preview",
#             messages=conversation,
#         )

#         # Extract and return the model's reply
#         reply = response.choices[0].message.content
#         st.write('Response:', reply)
#     else:
#         st.write('Please enter a query.')