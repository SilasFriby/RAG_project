import os
import json
import openai
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index import ServiceContext
import nest_asyncio
from llama_index.llms import OpenAI, HuggingFaceInferenceAPI, Ollama
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.text_splitter import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.embeddings import OpenAIEmbedding
from llama_index import Document
from llama_index.postprocessor import LLMRerank
from prompts.llm_re_ranker_prompt_template import CUSTOM_CHOICE_SELECT_PROMPT
from prompts.sub_question_prompt_template import CUSTOM_SUB_QUESTION_PROMPT_TMPL
from prompts.vector_store_query_prompt_template import CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL
from prompts.qa_prompt_template import CUSTOM_QUESTION_GEN_TMPL
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from custom_classes.custom_perplexity_llm import CustomPerplexityLLM
from llama_index.extractors import QuestionsAnsweredExtractor, KeywordExtractor
import logging
import sys
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.retrievers import KeywordTableSimpleRetriever
from custom_classes.custom_retriever import CustomRetriever
from llama_index import SimpleKeywordTableIndex
from llama_index.indices.prompt_helper import PromptHelper


# Initialize variables
documents_dir = "data/statements_txt_files"
documents_file_path = "data/test.jsonl"  # statements_id_title_text_sub.jsonl"
llm_model_name = "gpt-3.5-turbo-1106"
llm_temp = 0
llm_response_max_tokens = 1024
embed_model_name = "text-embedding-ada-002"  # "WhereIsAI/UAE-Large-V1" #"intfloat/e5-mistral-7b-instruct" #"BAAI/bge-large-en-v1.5" #"sentence-transformers/all-MiniLM-L6-v2"
top_k = 5
chunk_size = 512
chunk_overlap = 20
file_path_titles = "data/statements_id_title_sub.jsonl"
response_mode = "compact"
n_keywords = 5
n_qa = 3
llm_model_name_gold_standard = "gpt-4"
query_str = "Discuss the new ventures Caravelle International Group is planning to launch in 2023 and explain how they are expected to offset any potential weakness in their shipping business. Also, provide a brief overview of the financial performance of the company in 2022."

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# nested asynchronous requests
nest_asyncio.apply()

llm_model = OpenAI(
    model=llm_model_name,
    temperature=llm_temp,
    max_tokens=llm_response_max_tokens,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# LLM gold standard
llm_model_gold_standard = OpenAI(
    model=llm_model_name_gold_standard,
    temperature=llm_temp,
    max_tokens=llm_response_max_tokens,
    api_key=os.getenv("OPENAI_API_KEY"),
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
documents = [
    Document(
        text=info["text"],
        metadata={
            "title": info["title"],
            "document_id": info["id"],
            "company_name": info["company_name"],
            "year": info["year"],
        },
    )
    for info in documents_info
]

# print(documents[0].get_content(metadata_mode=MetadataMode.LLM))
# print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))

# QA extractor
qa_extractor = QuestionsAnsweredExtractor(
    llm=llm_model, 
    questions=n_qa, 
    prompt_template=CUSTOM_QUESTION_GEN_TMPL,
)
keyword_extractor = KeywordExtractor(
    llm=llm_model,
    keywords=n_keywords
)

# Chunking
text_splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Prompt helper - helps deal with LLM context window token limitations.
prompt_helper = PromptHelper()

# Service context
service_context = ServiceContext.from_defaults(
    llm=llm_model,
    embed_model=embed_model,
    prompt_helper=prompt_helper,
)

# Set the global service context
set_global_service_context(service_context)

# Logging in order to see API calls
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Chromadb
# chroma_client = chromadb.EphemeralClient()
# # chroma_client.delete_collection("quickstart")
# chroma_collection = chroma_client.create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# # Save Chromadb to disk
# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load Chromadb from disk
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Vector index
vector_index = VectorStoreIndex.from_vector_store(vector_store)

# Keyword index
nodes = text_splitter(documents)
keyword_index = SimpleKeywordTableIndex(
    nodes=nodes,
    service_context=service_context
)

# Metadata filters
vector_store_info = VectorStoreInfo(
    content_info="Financial statements",
    metadata_info=[
        MetadataInfo(
            name="company_name",
            type="str",
            description=(
                "The name of the company that published the financial statement, e.g. Digizuite"
            ),
        ),
    ],
)

# Custom retriever
vector_retriever = VectorIndexRetriever(
    index=vector_index, 
    similarity_top_k=top_k
)

keyword_retriever = KeywordTableSimpleRetriever(
    index=keyword_index,
    similarity_top_k=top_k,)

meta_filter_retriever = VectorIndexAutoRetriever(
    index=vector_index,
    vector_store_info=vector_store_info,
    similarity_top_k=top_k,
    prompt_template_str=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL,
    verbose=True,
)
custom_retriever = CustomRetriever(
    vector_retriever=vector_retriever, 
    keyword_retriever=keyword_retriever,
    meta_filter_retreiver=meta_filter_retriever,
    mode="OR")

# Respone synthesizer
response_synthesizer = get_response_synthesizer(response_mode=response_mode)

# Re-ranking of nodes - use custom prompt if not OpenAI LLM
if llm_model.api_key == os.getenv("OPENAI_API_KEY"):
    reranker = LLMRerank(
        service_context=service_context,
    )
else:
    reranker = LLMRerank(
        service_context=service_context,
        choice_select_prompt=CUSTOM_CHOICE_SELECT_PROMPT,
    )

# Base query engine
base_query_engine = RetrieverQueryEngine.from_args(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
    node_postprocessors=[reranker],
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

# Sub query engine - use custom question generator if not OpenAI LLM
if llm_model.api_key == os.getenv("OPENAI_API_KEY"):
    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        use_async=False,
        verbose=True,
    )
else:
    question_gen = LLMQuestionGenerator.from_defaults(
        prompt_template_str=CUSTOM_SUB_QUESTION_PROMPT_TMPL
    )

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        use_async=True,
        question_gen=question_gen,
        verbose=True,
    )


response = sub_query_engine.query(query_str)
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

# # Storage context
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

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

# API_URL = "https://ghprpg1pq3gveb5b.us-east-1.aws.endpoints.huggingface.cloud"

# elif choose_llm_model == 3:
#     llm_model = HuggingFaceInferenceAPI(
#         model_name=API_URL,
#         temperature=llm_temp,
#         max_tokens=llm_response_max_tokens,
#         token=os.getenv("HUGGING_FACE_TOKEN"),
#     )


# # Metadata filters
# vector_store_info = VectorStoreInfo(
#     content_info="Financial statements",
#     metadata_info=[
#         MetadataInfo(
#             name="company_name",
#             type="str",
#             description=(
#                 "The name of the company that published the financial statement, e.g. Digizuite"
#             ),
#         ),
#         # MetadataInfo(
#         #     name="year",
#         #     type="int",
#         #     description=(
#         #         "The year corresponding to the financial statement, e.g. 2023"
#         #     ),
#         # ),
#     ],
# )

# # Retriever
# retriever = VectorIndexAutoRetriever(
#     index=index,
#     vector_store_info=vector_store_info,
#     similarity_top_k=top_k,
#     prompt_template_str=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL,
# )

# query_str = "Tell me about UP Fintech in 2022" #"Discuss the new ventures Caravelle International Group in 2022"# is planning to launch in 2023 and explain how they are expected to offset any potential weakness in their shipping business."# Also, provide a brief overview of the financial performance of the company in 2022."

# from llama_index.vector_stores.types import VectorStoreQuerySpec
# from llama_index.prompts.base import PromptTemplate
# query_bundle = QueryBundle(query_str=query_str)
# info_str = vector_store_info.json(indent=4)
# schema_str = VectorStoreQuerySpec.schema_json(indent=4)
# prompt = PromptTemplate(template=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL)

# # call LLM
# output = service_context.llm.predict(
#     prompt,
#     schema_str=schema_str,
#     info_str=info_str,
#     query_str=query_bundle.query_str,
# )
# print(output)


# parse output
# parse_generated_spec(output, query_bundle)

# retriever._parse_generated_spec(QueryBundle(query_str=query_str))
# retriever.generate_retrieval_spec(QueryBundle(query_str=query_str))

# from llama_index.vector_stores.types import (
#     FilterOperator,
#     FilterCondition,
#     MetadataFilters,
#     MetadataFilter,
# )

# filters = MetadataFilters(
#     filters=[
#         MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE),
#         MetadataFilter(key="year", value=2023, operator=FilterOperator.LTE),
#         MetadataFilter(key="company_name", value="Caravelle International Group", operator=FilterOperator.EQ),
#     ],
# )

# query_str = "Tell me something" #"Discuss the new ventures Caravelle International Group in 2022"# is planning to launch in 2023 and explain how they are expected to offset any potential weakness in their shipping business."# Also, provide a brief overview of the financial performance of the company in 2022."


# retriever = index.as_retriever(filters=filters)
# retrieved_nodes = retriever.retrieve(query_str)
# print(retrieved_nodes[0].metadata)

# # Retriever evaluation
# import asyncio
# from llama_index.evaluation import generate_question_context_pairs
# from llama_index.evaluation import RetrieverEvaluator

# qa_dataset = generate_question_context_pairs(
#     nodes, 
#     llm=llm_model_gold_standard, 
#     num_questions_per_chunk=1
# )

# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=top_k,
# )


# retriever_evaluator = RetrieverEvaluator.from_metric_names(
#     ["mrr", "hit_rate"], retriever=retriever
# )

# async def retriever_evaluation(qa_dataset):
#     eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
#     return eval_results

# eval_results = asyncio.run(retriever_evaluation(qa_dataset))
# print(eval_results)

# ids = list(qa_dataset.queries.keys())
# i = 4
# print(qa_dataset.queries[ids[i]])
# for i, id in enumerate(ids):
#     print(i)
#     retriever_evaluator.evaluate(
#         query=qa_dataset.queries[ids[i]],
#         expected_ids=qa_dataset.relevant_docs[ids[i]])

# reranked_nodes = reranker.postprocess_nodes(
#     nodes=retrieved_nodes,
#     query_str=query_str
# )

# # Compare retrieved nodes and reranked nodes
# for node in retrieved_nodes:
#     print(node.node_id , node.metadata["title"])
# for node in reranked_nodes:
#     print(node.node_id , node.metadata["title"])

# base_query_engine = index.as_query_engine(
#     response_synthesizer=response_synthesizer, node_postprocessors=[reranker]
# )

# from llama_index.indices.utils import default_format_node_batch_fn, default_parse_choice_select_answer_fn
# nodes = vector_retriever.retrieve(query_str)
# query_bundle = QueryBundle(query_str=query_str)
# choice_batch_size = 10
# format_node_batch_fn = default_format_node_batch_fn
# parse_choice_select_answer_fn = default_parse_choice_select_answer_fn
# top_n = 10

# def postprocess_nodes(
#         nodes: List[NodeWithScore],
#         query_bundle: [QueryBundle] = None,
#     ) -> List[NodeWithScore]:
#         if query_bundle is None:
#             raise ValueError("Query bundle must be provided.")
#         if len(nodes) == 0:
#             return []

#         initial_results: List[NodeWithScore] = []
#         for idx in range(0, len(nodes), choice_batch_size):
#             print(idx)
#             nodes_batch = [
#                 node.node for node in nodes[idx : idx + choice_batch_size]
#             ]

#             query_str = query_bundle.query_str
#             fmt_batch_str = format_node_batch_fn(nodes_batch)
#             # call each batch independently
#             raw_response = service_context.llm.predict(
#                 CUSTOM_CHOICE_SELECT_PROMPT,
#                 context_str=fmt_batch_str,
#                 query_str=query_str,
#             )

#             raw_choices, relevances = parse_choice_select_answer_fn(
#                 raw_response, len(nodes_batch)
#             )

#             # answer = raw_response
#             # num_choices = len(nodes_batch)
#             # raise_error = False

            
#             # answer_lines = answer.split("\n")
#             # answer_nums = []
#             # answer_relevances = []
#             # for answer_line in answer_lines:
#             #     line_tokens = answer_line.split(",")
#             #     if len(line_tokens) != 2:
#             #         if not raise_error:
#             #             continue
#             #         else:
#             #             raise ValueError(
#             #                 f"Invalid answer line: {answer_line}. "
#             #                 "Answer line must be of the form: "
#             #                 "answer_num: <int>, answer_relevance: <float>"
#             #             )
#             #     answer_num = int(line_tokens[0].split(":")[1].strip())
#             #     if answer_num > num_choices:
#             #         continue
#             #     answer_nums.append(answer_num)
#             #     answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))

#             # raw_choices, relevances = answer_nums, answer_relevances

#             choice_idxs = [int(choice) - 1 for choice in raw_choices]
#             choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
#             relevances = relevances or [1.0 for _ in choice_nodes]
#             initial_results.extend(
#                 [
#                     NodeWithScore(node=node, score=relevance)
#                     for node, relevance in zip(choice_nodes, relevances)
#                 ]
#             )

#         return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
#             : top_n
#         ]

# postprocess_nodes(
#     nodes=nodes,
#     query_bundle=query_bundle
# )

# retrieved_nodes = vector_retriever.retrieve(query_str)
# reranked_nodes = reranker.postprocess_nodes(
#     nodes=retrieved_nodes,
#     query_str=query_str
# )

# # Compare retrieved nodes and reranked nodes
# for node in retrieved_nodes:
#     print(node.node_id , node.metadata["company_name"])
# for node in reranked_nodes:
#     print(node.node_id , node.metadata["company_name"])


# from llama_index.callbacks.schema import CBEventType, EventPayload
# from llama_index.utils import get_color_mapping, print_text
# from llama_index.async_utils import run_async_tasks
# from typing import List, Optional, Sequence, cast
# from llama_index.bridge.pydantic import BaseModel, Field
# from llama_index.question_gen.types import BaseQuestionGenerator, SubQuestion

# class SubQuestionAnswerPair(BaseModel):
#     """
#     Pair of the sub question and optionally its answer (if its been answered yet).
#     """

#     sub_q: SubQuestion
#     answer: Optional[str] = None
#     sources: List[NodeWithScore] = Field(default_factory=list)

# callback_manager = service_context.callback_manager
# metadatas = [x.metadata for x in query_engine_tools]
# verbose = True
# use_async = False
# query_bundle = QueryBundle(query_str=query_str)
# logger = logging.getLogger(__name__)
# query_engines = {
#             tool.metadata.name: tool.query_engine for tool in query_engine_tools
#         }
# node_postprocessors = [reranker]
# from llama_index.indices.utils import default_format_node_batch_fn, default_parse_choice_select_answer_fn
# format_node_batch_fn = default_format_node_batch_fn
# # parse_choice_select_answer_fn = default_parse_choice_select_answer_fn
# choice_batch_size = 10
# top_n = 10

# def parse_choice_select_answer_fn(
#     answer: str, num_choices: int, raise_error: bool = False
# ) -> Tuple[List[int], List[float]]:
#     """Default parse choice select answer function."""
#     answer_lines = answer.split("\n")
#     answer_nums = []
#     answer_relevances = []
#     for answer_line in answer_lines:
#         line_tokens = answer_line.split(",")
#         if len(line_tokens) != 2:
#             if not raise_error:
#                 continue
#             else:
#                 raise ValueError(
#                     f"Invalid answer line: {answer_line}. "
#                     "Answer line must be of the form: "
#                     "answer_num: <int>, answer_relevance: <float>"
#                 )
#         answer_num = int(line_tokens[0].split(":")[1].strip())
#         if answer_num > num_choices:
#             continue
#         answer_nums.append(answer_num)
#         answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
#     return answer_nums, answer_relevances

# def postprocess_nodes(
#         nodes: List[NodeWithScore],
#         query_bundle: Optional[QueryBundle] = None,
#     ):
#         if query_bundle is None:
#             raise ValueError("Query bundle must be provided.")
#         if len(nodes) == 0:
#             return []

#         initial_results: List[NodeWithScore] = []
#         for idx in range(0, len(nodes), choice_batch_size):
#             nodes_batch = [
#                 node.node for node in nodes[idx : idx + choice_batch_size]
#             ]

#             query_str = query_bundle.query_str
#             fmt_batch_str = format_node_batch_fn(nodes_batch)
#             # call each batch independently
#             raw_response = service_context.llm.predict(
#                 CUSTOM_CHOICE_SELECT_PROMPT,
#                 context_str=fmt_batch_str,
#                 query_str=query_str,
#             )

#             raw_choices, relevances = parse_choice_select_answer_fn(
#                 raw_response, len(nodes_batch)
#             )
#             choice_idxs = [int(choice) - 1 for choice in raw_choices]
#             choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
#             relevances = relevances or [1.0 for _ in choice_nodes]
#             initial_results.extend(
#                 [
#                     NodeWithScore(node=node, score=relevance)
#                     for node, relevance in zip(choice_nodes, relevances)
#                 ]
#             )

#         nodes_post = sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
#             : top_n
#         ]

#         return nodes_post

# def apply_node_postprocessors(
#        nodes: List[NodeWithScore], query_bundle: QueryBundle
#     ):
#         nodes = postprocess_nodes(
#             nodes, query_bundle=query_bundle
#         )
#         return nodes

# def retrieve(query_bundle: QueryBundle):
#     nodes = vector_retriever.retrieve(query_bundle)
#     nodes_post = apply_node_postprocessors(nodes, query_bundle=query_bundle)
#     return nodes_post

# def query1(query_bundle: QueryBundle):
#     """Answer a query."""
#     with callback_manager.event(
#         CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
#     ) as query_event1:
#         nodes = retrieve(query_bundle)
#         response = response_synthesizer.synthesize(
#             query=query_bundle,
#             nodes=nodes,
#         )

#         query_event1.on_end(payload={EventPayload.RESPONSE: response})

#     return response

# def query_subq(
#         sub_q: SubQuestion, color: Optional[str] = None
#     ) -> Optional[SubQuestionAnswerPair]:
#         try:
#             with callback_manager.event(
#                 CBEventType.SUB_QUESTION,
#                 payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
#             ) as event:
#                 question = sub_q.sub_question
#                 query_engine = query_engines[sub_q.tool_name]

#                 if verbose:
#                     print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

#                 response = query_engine.query(QueryBundle(query_str=question))
#                 # response = query1(QueryBundle(query_str=question))
#                 response_text = str(response)

#                 if verbose:
#                     print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

#                 qa_pair = SubQuestionAnswerPair(
#                     sub_q=sub_q, answer=response_text, sources=response.source_nodes
#                 )

#                 event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

#             return qa_pair
#         except ValueError:
#             logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
#             return None

#                 # with callback_manager.event(
#                 #     CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
#                 # ) as query_event:
#                 #     nodes = vector_retriever.retrieve(query_bundle)
#                     # initial_results: List[NodeWithScore] = []
#                     # for idx in range(0, len(nodes), 10):
#                     #     nodes_batch = [
#                     #         node.node for node in nodes[idx : idx + 10]
#                     #     ]

#                     #     query_str = query_bundle.query_str
#                     #     fmt_batch_str = format_node_batch_fn(nodes_batch)
#                     #     # call each batch independently
#                     #     # raw_response = service_context.llm.predict(
#                     #     #     CUSTOM_CHOICE_SELECT_PROMPT,
#                     #     #     context_str=fmt_batch_str,
#                     #     #     query_str=query_str,
#                     #     # )

#                     #     raw_response = "Doc: 1, Relevance: 10\nDoc: 2, Relevance: 8\n\nDocument 1 provides the exact revenue figure for UP Fintech in Q2 2023, which is the most direct answer to the user's question. Document 2, while not providing the exact revenue figure, offers additional context and confirms the revenue growth, as well as other related financial information"

#                     #     raw_choices, relevances = parse_choice_select_answer_fn(
#                     #         raw_response, len(nodes_batch)
#                     #     )
#                     #     choice_idxs = [int(choice) - 1 for choice in raw_choices]
#                     #     choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
#                     #     relevances = relevances or [1.0 for _ in choice_nodes]
#                     #     initial_results.extend(
#                     #         [
#                     #             NodeWithScore(node=node, score=relevance)
#                     #             for node, relevance in zip(choice_nodes, relevances)
#                     #         ]
#                     #     )

#                     # nodes = sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
#                     #     : 10
#                     # ]

#         #             response = response_synthesizer.synthesize(
#         #                 query=query_bundle,
#         #                 nodes=nodes,
#         #             )

#         #             query_event.on_end(payload={EventPayload.RESPONSE: response})

#         #         response_text = str(response)

#         #         if verbose:
#         #             print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

#         #         qa_pair = SubQuestionAnswerPair(
#         #             sub_q=sub_q, answer=response_text, sources=response.source_nodes
#         #         )

#         #         event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

#         #     return qa_pair
#         # except ValueError:
#         #     logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
#         #     return None

# def query(query_bundle: QueryBundle):
#     with callback_manager.event(
#         CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
#     ) as query_event:
#         sub_questions = question_gen.generate(metadatas, query_bundle)

#         colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

#         if verbose:
#             print_text(f"Generated {len(sub_questions)} sub questions.\n")

#         if use_async:
#             tasks = [
#                 sub_query_engine._aquery_subq(sub_q, color=colors[str(ind)])
#                 for ind, sub_q in enumerate(sub_questions)
#             ]
#             print(tasks)
#             qa_pairs_all = run_async_tasks(tasks)
#             qa_pairs_all = cast(List[Optional[SubQuestionAnswerPair]], qa_pairs_all)
#         else:
#             qa_pairs_all = [
#                 query_subq(sub_q, color=colors[str(ind)])
#                 for ind, sub_q in enumerate(sub_questions)
#             ]

#         # filter out sub questions that failed
#         qa_pairs: List[SubQuestionAnswerPair] = list(filter(None, qa_pairs_all))

#         nodes = [sub_query_engine._construct_node(pair) for pair in qa_pairs]

#         source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]
#         response = response_synthesizer.synthesize(
#             query=query_bundle,
#             nodes=nodes,
#             additional_source_nodes=source_nodes,
#         )

#         query_event.on_end(payload={EventPayload.RESPONSE: response})

#     return response


# # response = query(query_bundle)
# # print(response)

# # Ingestion pipeline
# pipeline = IngestionPipeline(
#     transformations=[
#         text_splitter,
#         # qa_extractor,
#         # keyword_extractor,
#         embed_model,
#     ],
#     vector_store=vector_store,
# )

# # Ingest directly into a vector db
# nodes = pipeline.run(
#     documents=documents, 
#     show_progress=True
# )


# llm_model_names = [
#     "llama-2-70b-chat",
#     "mixtral-8x7b-instruct",
# ] 
# LLM
# llm_model = CustomPerplexityLLM(
#     model=llm_model_names[choose_llm_model],
#     temperature=llm_temp,
#     max_tokens=llm_response_max_tokens,
#     api_key=os.getenv("PERPLEXITY_API_KEY"),
# )
# llm_model = Ollama(
#     model="zephyr",
#     temperature=llm_temp,
#     max_tokens=llm_response_max_tokens,
# )