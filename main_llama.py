import openai
import os
import json
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex #, ServiceContext, PromptHelper
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.ingestion import IngestionPipeline


# Initialize variables
documents_dir = "data/statements_txt_files"
llm_model = "gpt-3.5-turbo-0613"
llm_response_max_tokens = 256
llm_temp = 0
chunk_size = 1024
chunk_overlap = 0
paragraph_separator = "\n\n"
system_prompt = "Hello, I am a financial analyst. My expertise is answering questions about financial statements."


# OpenAI API
# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# LLM
llm = OpenAI(model=llm_model, temperature=llm_temp, max_tokens=llm_response_max_tokens)

# Read the documents from the directory
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
documents = reader.load_data()

# Chunking
text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    paragraph_separator=paragraph_separator,
)

# Meta data extractors
title_extractor = TitleExtractor(nodes=1, llm=llm) #title is located on the first page, so pass 1 to nodes param
qa_extractor = QuestionsAnsweredExtractor(questions=3), #let's extract 3 questions for each node, you can customize this.
summary_extractor = SummaryExtractor(summaries=["prev", "self"], llm=llm), #let's extract the summary for both previous node and current node.
key_word_extractor = KeywordExtractor(keywords=10, llm=llm) #let's extract 10 keywords for each node.

# Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create ingestion pipeline
transformations = [
    SentenceSplitter(),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
    SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10),
]

pipeline = IngestionPipeline(
    transformations=transformations,)

# Create nodes
nodes = pipeline.run(documents=documents, show_progress=True)

# Print metadata in json format
for node in nodes:
    metadata_json = json.dumps(node.metadata, indent=4)  # Convert metadata to formatted JSON
    print(metadata_json)

# Indexing
index = VectorStoreIndex(nodes)


# Create chat engine that uses the index
chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

response = chat_engine.chat("Hi")
print(response)

response = chat_engine.chat(
    "What was the revenue for UP Fintech in the second quater of 2023?",
    tool_choice="query_engine_tool",
)
print(response)


# # Prompt Helper
# prompt_helper = PromptHelper()

# # Service Context - a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline
# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
#     text_splitter=text_splitter,
#     prompt_helper=prompt_helper,
#     system_prompt=system_prompt,
#     transformations=[text_splitter, title_extractor, qa_extractor, summary_extractor, key_word_extractor],
# )

# # Indexing
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context, show_progress=True
# )
