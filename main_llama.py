import openai
import os
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    PromptHelper,
    load_index_from_storage,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter

# from llama_index.extractors import (
#     #SummaryExtractor,
#     #QuestionsAnsweredExtractor,
#     TitleExtractor,
#     #KeywordExtractor,
# )
from llama_index.llms import HuggingFaceInferenceAPI
import torch
from llama_index.llms import Ollama
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
import nest_asyncio
from llama_index.node_parser import SimpleNodeParser


# Initialize variables
documents_dir = "data/statements_txt_files"
llm_gpt3_name = "gpt-3.5-turbo-0613"
llm_llama2_name = "llama2"
llm_hf_name = "berkeley-nest/Starling-LM-7B-alpha"
llm_response_max_tokens = 1024
llm_temp = 0
# chunk_size = 256
chunk_overlap = 0
paragraph_separator = "\n\n"
# system_prompt = "Hello, I am a financial analyst. My expertise is answering questions about financial statements."
save_index = False
load_index = False
# custom_title_prompt = """\
# Context: {context_str}. Give a title that summarizes all of \
# the unique entities, titles or themes found in the context. Your answer should consist of nothing else but this title. Title: """

# nested asynchronous requests
nest_asyncio.apply()

# # Load environment variables from .env file
# load_dotenv()

# # Hugging Face Token
# # HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# # Get the OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Set the OpenAI API key
# openai.api_key = openai_api_key

# # LLMs
# llm_gpt3 = OpenAI(model=llm_gpt3_name, temperature=llm_temp, max_tokens=llm_response_max_tokens)
llm_llama2 = Ollama(
    model=llm_llama2_name, temperature=llm_temp, max_tokens=llm_response_max_tokens
)


# Read the documents from the directory
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
documents = reader.load_data()

# Parse nodes
node_parser = SimpleNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

# # Chunking
# text_splitter = SentenceSplitter(
#     # chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     paragraph_separator=paragraph_separator,
# )

# # Meta data extractors
# meta_data_extractors = [
#     TitleExtractor(nodes=1, llm=llm_llama2, node_template=custom_title_prompt),
#     # QuestionsAnsweredExtractor(questions=3, llm=llm_hf),
#     # SummaryExtractor(summaries=["prev", "self"], llm=llm_llama2),
#     # KeywordExtractor(keywords=10, llm=llm_hf),
# ]

# Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Prompt Helper
# prompt_helper = PromptHelper()

# Service Context - a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline
service_context = ServiceContext.from_defaults(
    llm=llm_llama2,
    embed_model=embed_model,
    # text_splitter=text_splitter,
    # prompt_helper=prompt_helper,
    # system_prompt=system_prompt,
    # transformations=[text_splitter],  # + meta_data_extractors,
)


# Set the global service context
set_global_service_context(service_context)


# Indexing - load from disk or create new index from documents
if load_index:
    storage_context = StorageContext.from_defaults(persist_dir="vector_store")
    index = load_index_from_storage(storage_context, index_id="vector_index")
else:
    index = VectorStoreIndex.from_documents(documents, use_async=True, show_progress=True)

# Save index to disk
if save_index:
    index.set_index_id("vector_index")
    index.storage_context.persist("./vector_store")

# # See meta data
# index.ref_doc_info


# Configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

# Configure response synthesizer - when used in a query engine, the response synthesizer is used after nodes are retrieved from a retriever, and after any node-postprocessors are ran.
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",  # see https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html
    structured_answer_filtering=True, # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html
)

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)],
)

# response = query_engine.query("what is the revenue for UP Fintech?")# and Top Strike")

# print(response)

from llama_index.llama_pack import download_llama_pack

QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack",
    "./query_rewriting_pack",
    # leave the below commented out (was for testing purposes)
    # llama_hub_url="https://raw.githubusercontent.com/run-llama/llama-hub/jerry/add_llama_packs/llama_hub",
)
query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,
    chunk_size=256,
    vector_similarity_top_k=2,
)

# this will run the full pack
response = query_rewriting_pack.run("Compare revenues for UP Fintech and Top Strike")
print(str(response))



# remotely_run = HuggingFaceInferenceAPI(
#     model_name="berkeley-nest/Starling-LM-7B-alpha",
#     token=HF_TOKEN,
#     url=API_URL,
# )

# message = [ChatMessage(
#     role="user",  # Role can be 'user' or 'system'
#     content="What is the capital of France?"  # The actual message content
# )]

# response = remotely_run.chat(message)
# print(response)

# HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# # This is your custom Hugging Face Inference Endpoint URL
# ENDPOINT_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"

# # Initialize the HuggingFaceInferenceAPI with your custom endpoint
# llm_hf = HuggingFaceInferenceAPI(
#     model_name=ENDPOINT_URL,
#     token=HF_TOKEN,
#     # Specify any additional parameters you might need
# )

# message = [ChatMessage(
#     role="user",  # Role can be 'user' or 'system'
#     content="What is the capital of France?"  # The actual message content
# )]

# response = llm_hf.chat(message)
# print(response)

# Ollama
