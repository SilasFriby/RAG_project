
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

import nest_asyncio

nest_asyncio.apply()
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(
    callback_manager=callback_manager
)

# load data
documents_dir = "data/statements_txt_files"
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
documents = reader.load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    documents, use_async=True, service_context=service_context
).as_query_engine()

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="documents",
            description="Financial statements",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)

response = query_engine.query(
     "What was the revenues for UP Fintech and Top Strike?"
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