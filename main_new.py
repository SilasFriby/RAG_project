import os
import json
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext
import nest_asyncio
from llama_index.llms import Ollama, OpenAI, HuggingFaceInferenceAPI#, Perplexity 
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage import StorageContext
from llama_index.schema import BaseNode, NodeRelationship
from typing import List
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.text_splitter import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import QueryBundle
from llama_index.embeddings import OpenAIEmbedding
from llama_index import Document
import pinecone
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.postprocessor import LLMRerank
from llama_index.schema import NodeWithScore, QueryBundle
from typing import Tuple
from llama_index.schema import BaseNode, MetadataMode
from prompts.llm_re_ranker_prompt_template import CUSTOM_CHOICE_SELECT_PROMPT
from prompts.sub_question_prompt_template import CUSTOM_SUB_QUESTION_PROMPT_TMPL
from prompts.vector_store_query_prompt_template import CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from custom_classes.custom_perplexity_llm import CustomPerplexityLLM


# Initialize variables
documents_dir = "data/statements_txt_files"
documents_file_path = "data/test.jsonl" #statements_id_title_text_sub.jsonl"
llm_model_names = ["llama2", "gpt-3.5-turbo-0613", "mixtral-8x7b-instruct"]#"mistral-7b-instruct"] #"codellama-34b-instruct"]
llm_temp = 0
llm_response_max_tokens = 1024
choose_llm_model = 2
API_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"
embed_model_name = "text-embedding-ada-002" #"WhereIsAI/UAE-Large-V1" #"intfloat/e5-mistral-7b-instruct" #"BAAI/bge-large-en-v1.5" #"sentence-transformers/all-MiniLM-L6-v2"
top_k = 10
chunk_size = 512
chunk_overlap = 20
file_path_titles = "data/statements_id_title_sub.jsonl"
query_str = "What was the revenue for UP Fintech?" #"What was the revenues for UP Fintech and Top Strike?"

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# nested asynchronous requests
nest_asyncio.apply()

# LLM
if choose_llm_model == 0:
    llm_model= Ollama(
        model=llm_model_names[choose_llm_model], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens
    )
elif choose_llm_model == 1:
    llm_model = OpenAI(
        model=llm_model_names[choose_llm_model], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("OPENAI_API_KEY")
    )
elif choose_llm_model == 2:
    llm_model = CustomPerplexityLLM(
        model=llm_model_names[choose_llm_model],
        temperature=llm_temp,
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("PERPLEXITY_API_KEY"), 
    )
elif choose_llm_model == 3:
    llm_model = HuggingFaceInferenceAPI(
        model_name=API_URL, 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )

from llama_index.llms import ChatMessage

# messages_dict = [
#     {"role": "system", "content": "Be precise and concise."},
#     {"role": "user", "content": "Tell me 5 sentences about Perplexity."},
# ]
# messages = [ChatMessage(**msg) for msg in messages_dict]
# llm_model.chat(messages)

# Embedding
# embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
embed_model = OpenAIEmbedding(model=embed_model_name)

# Read the documents from the directory
# reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
# docs = reader.load_data()

# Read jsonl file from documents file path
documents_info = []
with open(documents_file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        documents_info.append(data)


# Metadata
# Create Document objects with titles in their metadata
# documents = [Document(text=info['text'], metadata={'company_name': info['company_name'], 'title': info['title'], 'document_id': info['id']}) for info in documents_info]
documents = [Document(text=info['text'], metadata={'title': info['title'], 'document_id': info['id']}) for info in documents_info]


# from llama_index.schema import MetadataMode
# print(documents[0].get_content(metadata_mode=MetadataMode.LLM))
# print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# Chunking
text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Service context
service_context = ServiceContext.from_defaults(
    llm=llm_model,
    embed_model=embed_model,
    text_splitter=text_splitter,
)

# Set the global service context
set_global_service_context(service_context)

# # Initialize Pinecone 
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"), 
#     environment="gcp-starter")
# pinecone.delete_index( "ragindex")
# pinecone_index = pinecone.create_index(
#     "ragindex", 
#     dimension=1536, 
#     metric="dotproduct", 
#     pod_type="p1"
# )

# # Vector store Pinecone - set add_sparse_vector=True to compute sparse vectors during upsert
# vector_store = PineconeVectorStore(
#     pinecone_index=pinecone_index,
#     add_sparse_vector=True,
#     index_name="ragindex",
#     environment="gcp-starter",
# )


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import chromadb
from llama_index.vector_stores import ChromaVectorStore

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index
index = VectorStoreIndex.from_documents(
    documents=documents, 
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)

# Metadata filters
vector_store_info = VectorStoreInfo(
    content_info="Financial statements",
    metadata_info=[
        MetadataInfo(
            name="title",
            type="str",
            description=(
                "The title of the financial statement, hence a full sentence, such as for example 'Digizuite publishes the half-year report for 2023-H1'"
            ),
        ),
        MetadataInfo(
            name="document_id",
            type="str",
            description=(
                "A unique identifier for the financial statement, for example b2fc1032799f655b"
            ),
        ),
        # MetadataInfo(
        #     name="company_name",
        #     type="str",
        #     description=(
        #         "The name of the company that published the financial statement, e.g. Digizuite"
        #     ),
        # ),
    ],
)

# Retriever
retriever = VectorIndexAutoRetriever(
    index=index,
    vector_store_info=vector_store_info,
    similarity_top_k=top_k,
    prompt_template_str=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL,
)

# # Retriver
# retriever = VectorIndexRetriever(
#         index=index,
#         similarity_top_k=top_k,
# )

# Retrieved nodes
retrieved_nodes = retriever.retrieve("Tell me about UP Fintech")
print(retrieved_nodes[0].metadata)
len(retrieved_nodes)

# Re-ranking - HOW TO SETUP USING RE-RANKED NODES????
reranker = LLMRerank(
            choice_batch_size=5,
            top_n=5,
            service_context=service_context,
            choice_select_prompt=CUSTOM_CHOICE_SELECT_PROMPT,
)
reranked_nodes = reranker.postprocess_nodes(
    nodes=retrieved_nodes, 
    query_str=query_str
)

# Compare retrieved nodes and reranked nodes
for node in retrieved_nodes:
    print(node.node_id , node.metadata["title"])
for node in reranked_nodes:
    print(node.node_id , node.metadata["title"])

# Query engine
query_engine = index.as_query_engine()#, reranked_nodes=reranked_nodes)
response = query_engine.query(query_str)
print(response)



from llama_index.retrievers import BM25Retriever
new_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=top_k)
new_retrieved_nodes = new_retriever.retrieve(QueryBundle(query_str))


for i, node in enumerate(retrieved_nodes):
    includes_firm = "UP Fintech" in node.text
    if includes_firm:
        print("NODE " + str(i) + "\n\n" + str(node.score) + "\n\n" + node.text + "\n\n")
 
for i, node in enumerate(new_retrieved_nodes):
    # node = new_retrieved_nodes[0]
    includes_firm = "UP Fintech" in node.text
    if includes_firm:
        print("NODE "  + str(i) + "\n\n" + str(node.score) + "\n\n" + node.text + "\n\n")


# Base query engine - 
base_query_engine = index.as_query_engine(retriever=base_retriever)

# Set up base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=base_query_engine,
        metadata=ToolMetadata(
            name="Financial statements",
            description="Financial information on companies",
        ),
    ),
]

# Question generator
question_gen = LLMQuestionGenerator.from_defaults(
    prompt_template_str= CUSTOM_SUB_QUESTION_PROMPT_TMPL
)

# Sub query engine
sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
    question_gen=question_gen,
)

response = sub_query_engine.query(
     query_str
)

print(response)


# from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo

# vector_store_info = VectorStoreInfo(
#     content_info="Financial statements",
#     metadata_info=[
#          MetadataInfo(
#             name="title",
#             description="A title for each financial statement",
#             type="string",
#         ),
#         MetadataInfo(
#             name="document_id",
#             description="A unique identifier for each financial statement",
#             type="string",
#         ),
#     ]
# )

# # Define retriever
# from llama_index.retrievers import VectorIndexAutoRetriever

# retriever = VectorIndexAutoRetriever(
#     index,
#     vector_store_info=vector_store_info,
#     similarity_top_k=2,
#     verbose=True,
# )

# retrieved_nodes = retriever.retrieve('What is the topic of the document with document_id 9a211f7b268c1ade?')
# retrieved_nodes[0].metadata


# retrieved_nodes = retriever.retrieve('What is the topic of the document with document_id 9a211f7b268c1ade?')
# retrieved_nodes[0].metadata

# for node in retrieved_nodes:
#     print(node.metadata["title"])