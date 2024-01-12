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
from llama_index.schema import MetadataMode
from prompts.llm_re_ranker_prompt_template import CUSTOM_CHOICE_SELECT_PROMPT
from prompts.sub_question_prompt_template import CUSTOM_SUB_QUESTION_PROMPT_TMPL
from prompts.vector_store_query_prompt_template import CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL
from prompts.qa_prompt_template import CUSTOM_QUESTION_GEN_TMPL
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from custom_classes.custom_perplexity_llm import CustomPerplexityLLM
from llama_index.extractors import QuestionsAnsweredExtractor, KeywordExtractor
from llama_index.ingestion import IngestionPipeline
import logging
import sys
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.response_synthesizers import get_response_synthesizer



# Initialize variables
documents_dir = "data/statements_txt_files"
documents_file_path = "data/test.jsonl" #statements_id_title_text_sub.jsonl"
llm_model_names = ["llama2", "gpt-3.5-turbo-0613", "mixtral-8x7b-instruct"]#"mistral-7b-instruct"]
llm_temp = 0
llm_response_max_tokens = 1024
choose_llm_model = 2
API_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"
embed_model_name = "text-embedding-ada-002" #"WhereIsAI/UAE-Large-V1" #"intfloat/e5-mistral-7b-instruct" #"BAAI/bge-large-en-v1.5" #"sentence-transformers/all-MiniLM-L6-v2"
top_k = 10
chunk_size = 512
chunk_overlap = 20
file_path_titles = "data/statements_id_title_sub.jsonl"
response_mode = "compact"
n_keywords = 5
n_qa = 3
query_str = "What was the revenues for UP Fintech and Top Strike?"

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


# Embedding
# embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
embed_model = OpenAIEmbedding(model=embed_model_name)

# Read documents 
documents_info = []
with open(documents_file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        documents_info.append(data)


# Metadata
# Create Document objects with titles in their metadata
# documents = [Document(text=info['text'], metadata={'company_name': info['company_name'], 'title': info['title'], 'document_id': info['id']}) for info in documents_info]
documents = [Document(text=info['text'], metadata={'title': info['title'], 'document_id': info['id']}) for info in documents_info]
# documents = documents[:2]

# QA extractor
qa_extractor = QuestionsAnsweredExtractor(questions=n_qa, prompt_template=CUSTOM_QUESTION_GEN_TMPL)
keyword_extractor = KeywordExtractor(keywords=n_keywords)

# Chunking
text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Service context
service_context = ServiceContext.from_defaults(
    llm=llm_model,
    embed_model=embed_model,
    transformations=[text_splitter, qa_extractor],
)

# Set the global service context
set_global_service_context(service_context)


# Logging in order to see API calls
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Chromadb
chroma_client = chromadb.EphemeralClient()
# chroma_client.delete_collection("quickstart")
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        text_splitter, 
        qa_extractor,
        keyword_extractor,
        embed_model,
    ],
    vector_store=vector_store,
)

# Nodes
nodes = pipeline.run(documents=documents)
# nodes[1].metadata
# nodes[1].text
# print(nodes[0].get_content(metadata_mode=MetadataMode.LLM))
# print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# Index
index = VectorStoreIndex.from_vector_store(vector_store)

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

# Retrieved nodes
retrieved_nodes = retriever.retrieve("Tell me about UP Fintech")
print(retrieved_nodes[0].metadata)
len(retrieved_nodes)


# Re-ranking - HOW TO SETUP USING RE-RANKED NODES???? Postprocessing nodes?
reranker = LLMRerank(
            choice_batch_size=5,
            top_n=5,
            service_context=service_context,
            choice_select_prompt=CUSTOM_CHOICE_SELECT_PROMPT,
)
# reranked_nodes = reranker.postprocess_nodes(
#     nodes=retrieved_nodes, 
#     query_str=query_str
# )

# # Compare retrieved nodes and reranked nodes
# for node in retrieved_nodes:
#     print(node.node_id , node.metadata["title"])
# for node in reranked_nodes:
#     print(node.node_id , node.metadata["title"])

# Respone synthesizer
response_synthesizer = get_response_synthesizer(response_mode=response_mode)

# Base query engine
base_query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker]
)

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

# # Vector store Pinecone - set add_sparse_vector=True to compute sparse vectors during upsert (used for keyword search)
# vector_store = PineconeVectorStore(
#     pinecone_index=pinecone_index,
#     add_sparse_vector=True,
#     index_name="ragindex",
#     environment="gcp-starter",
# )

# index = VectorStoreIndex.from_documents(
#     documents=documents, 
#     storage_context=storage_context,
#     service_context=service_context,
#     show_progress=True,
# )

# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=top_k,
# )