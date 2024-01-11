
import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext
import nest_asyncio
from llama_index.llms import Ollama, OpenAI, HuggingFaceInferenceAPI, Perplexity 
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from prompts.sub_question_prompt_template import CUSTOM_SUB_QUESTION_PROMPT_TMPL

# Initialize variables
documents_dir = "data/statements_txt_files"
llm_model_names = ["llama2", "gpt-3.5-turbo-0613", "mistral-7b-instruct"]
llm_temp = 0
llm_response_max_tokens = 1024
llm_model_index = 0
API_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
query_str = "What was the revenues for UP Fintech?" #"What was the revenues for UP Fintech and Top Strike?"


# Load environment variables from .env file
load_dotenv()

# LLM
if llm_model_index == 0:
    llm_model= Ollama(
        model=llm_model_names[llm_model_index], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens
    )
elif llm_model_index == 1:
    llm_model = OpenAI(
        model=llm_model_names[llm_model_index], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("OPENAI_API_KEY")
    )
elif llm_model_index == 2:
    llm_model = Perplexity(
        model=llm_model_names[llm_model_index],
        temperature=llm_temp,
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("PERPLEXITY_API_KEY"), 
    )
elif llm_model_index == 3:
    llm_model = HuggingFaceInferenceAPI(
        model_name=API_URL, 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )


# Embedding
embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

# nested asynchronous requests
nest_asyncio.apply()

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    llm=llm_model,
    embed_model=embed_model,
    callback_manager=callback_manager
)

# Set the global service context
set_global_service_context(service_context)

# Load data
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
documents = reader.load_data()

# Index
index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context,
    use_async=True
)

# Base query engine
base_query_engine = index.as_query_engine()

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




# import logging
# import sys
# from dotenv import load_dotenv
# import os
# import openai

# # Load environment variables from .env file
# load_dotenv()

# # Get the OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Set the OpenAI API key
# openai.api_key = openai_api_key

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# from llama_index import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     ServiceContext,
# )
# from llama_index.llms import OpenAI

# # LLM (gpt-3)
# gpt3 = OpenAI(temperature=0, model="text-davinci-003")
# service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)

# # LLM (gpt-4)
# gpt4 = OpenAI(temperature=0, model="gpt-4")
# service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# # load documents
# documents_dir = "data/statements_txt_files"
# reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
# documents = reader.load_data()
# index = VectorStoreIndex.from_documents(documents)

# from llama_index.indices.query.query_transform.base import (
#     StepDecomposeQueryTransform,
# )

# # gpt-4
# step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)

# # gpt-3
# step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
#     llm=gpt3, verbose=True
# )
# index_summary = "Used to answer questions about financial statements."

# # set Logging to DEBUG for more detailed outputs
# from llama_index.query_engine.multistep_query_engine import (
#     MultiStepQueryEngine,
# )

# query_engine = index.as_query_engine(service_context=service_context_gpt4)
# query_engine = MultiStepQueryEngine(
#     query_engine=query_engine,
#     query_transform=step_decompose_transform,
#     index_summary=index_summary,
# )
# response_gpt4 = query_engine.query(
#     "What was the revenues for UP Fintech and Top Strike?",
# )
# print(response_gpt4)
# sub_qa = response_gpt4.metadata["sub_qa"]
# tuples = [(t[0], t[1].response) for t in sub_qa]
# print(tuples)

# query_engine = index.as_query_engine(service_context=service_context_gpt3)
# query_engine = MultiStepQueryEngine(
#     query_engine=query_engine,
#     query_transform=step_decompose_transform_gpt3,
#     index_summary=index_summary,
# )

# response_gpt3 = query_engine.query(
#     "What was the revenues for UP Fintech and Top Strike?",
# )
# print(response_gpt3)
# sub_qa = response_gpt3.metadata["sub_qa"]
# tuples = [(t[0], t[1].response) for t in sub_qa]
# print(tuples)