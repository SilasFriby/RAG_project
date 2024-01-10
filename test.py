import logging
import sys
import pinecone
import os
import openai
from llama_index import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from dotenv import load_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# Pinecone index
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
pinecone_index = pinecone.create_index(
    "ragindex", dimension=1536, metric="dotproduct", pod_type="p1"
)
pinecone.describe_index("ragindex")

# set add_sparse_vector=True to compute sparse vectors during upsert
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
    index_name="ragindex",
    environment="gcp-starter",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine(similarity_top_k = 10)
response = query_engine.query("What is the topic of the document with document_id 9a211f7b268c1ade?")
print(response)