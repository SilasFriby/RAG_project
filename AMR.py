from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes, 
    get_root_nodes
)
from llama_index.llms import Ollama
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage import StorageContext
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.schema import BaseNode, NodeRelationship
from typing import List


# Initialize variables
documents_dir = "data/statements_txt_files"
llm_model_name = "llama2"
llm_temp = 0
llm_response_max_tokens = 1024
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
top_k = 10
AMR_chunk_sizes = [2048, 512, 128]
AMR_chunk_overlap = 20



# LLM
llm_model= Ollama(
    model=llm_model_name, temperature=llm_temp, max_tokens=llm_response_max_tokens
)

# Embedding
embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

# Read the documents from the directory
reader = SimpleDirectoryReader(input_dir=documents_dir, filename_as_id=True)
docs = reader.load_data()

# Create nodes
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=AMR_chunk_sizes, 
    chunk_overlap=AMR_chunk_overlap,
    include_metadata=True)

nodes = node_parser.get_nodes_from_documents(docs)
len(nodes)

# Simple helper function for fetching “intermediate” nodes within a node list. These are nodes that have both children and parent nodes.
# llama index only created helper functions for leaf nodes and root nodes, hence the need for this function.
def get_intermediate_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get intermediate nodes."""
    intermediate_nodes = []
    for node in nodes:
        if NodeRelationship.PARENT in node.relationships and NodeRelationship.CHILD in node.relationships:
            intermediate_nodes.append(node)
    return intermediate_nodes

# Get leaf, intermediate and root nodes
leaf_nodes = get_leaf_nodes(nodes)
intermediate_nodes = get_intermediate_nodes(nodes)
root_nodes = get_root_nodes(nodes)

len(leaf_nodes)
len(intermediate_nodes)
len(root_nodes)

# Add metadata to root nodes
for node in root_nodes:
    title = node.metadata['file_name']
    title = title.replace(".txt", "")
    title = title.replace("_", " ")
    node.metadata['title'] = title

# from llama_index.extractors import KeywordExtractor
# extractor = KeywordExtractor(llm=llm_model, keywords=2)
# metadata_dicts = extractor.extract(nodes[:2])
    
# Let leaf and intermediate nodes inherit metadata from their parent nodes
for node in intermediate_nodes:
    parent_id = node.parent_node.node_id
    matching_parent_node = [node for node in root_nodes if node.node_id == parent_id][0]
    parent_metadata = matching_parent_node.metadata
    node.metadata['title'] = parent_metadata['title']

for node in leaf_nodes:
    parent_id = node.parent_node.node_id
    matching_parent_node = [node for node in intermediate_nodes if node.node_id == parent_id][0]
    parent_metadata = matching_parent_node.metadata
    node.metadata['title'] = parent_metadata['title']

# Create docstore
docstore = SimpleDocumentStore()

# Insert nodes into docstore
docstore.add_documents(nodes)

# Define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)

# Define service context
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm=llm_model
)

# Load index into vector index
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    service_context=service_context,
)

# Define retriever
base_retriever = base_index.as_retriever(similarity_top_k=top_k)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# Query
query_1 = "What is the revenue for Top Strike?"
query_2 = "What are the revenues for UP Fintech and Top Strike."
query = query_2

# Display retrived nodes
nodes = retriever.retrieve(query)
base_nodes = base_retriever.retrieve(query)

# len(nodes)
# len(base_nodes)

# for i, node in enumerate(nodes):
#     print("NODE " + str(i) + "\n\n" + str(node.score) + "\n\n" + node.text + "\n\n")

# for i, node in enumerate(base_nodes):
#     includes_Top_Strike = True #node.metadata["file_name"] == "Top_Strike.txt"
#     if includes_Top_Strike:
#         print("NODE " + str(i) + "\n\n" + str(node.score) + "\n\n" + node.metadata["file_name"] + "\n\n" + node.text + "\n\n")


# Query engine
query_engine = RetrieverQueryEngine.from_args(retriever, service_context=service_context)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever, service_context=service_context)

response = query_engine.query(query)
base_response = base_query_engine.query(query)

print("AMR:" + "\n\n" + str(response) + "\n\n")
print("BASE:" + "\n\n" + str(base_response))

# from llama_index.question_gen.llm_generators import LLMQuestionGenerator
# from llama_index.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL
# from llama_index.tools.types import ToolMetadata
# from llama_index.schema import QueryBundle


# question_gen = LLMQuestionGenerator.from_defaults(
#     service_context=service_context,
#     prompt_template_str="""
#         Follow the example, but instead of giving a question, always prefix the question 
#         with: 'By first identifying and quoting the most relevant sources, '. 
#         """
#     + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
# )

# question_gen.generate(
#     tools=[
#         ToolMetadata(
#             name="Financial statements",
#             description="Financial information on companies",
#         )
#     ],
#     query=QueryBundle(query_str=query),
# )

# question_gen._get_prompts()

# from llama_index import VectorStoreIndex
# from llama_index.query_engine import SubQuestionQueryEngine
# from llama_index.tools import QueryEngineTool, ToolMetadata

# index = VectorStoreIndex(
#     nodes=nodes,
#     service_context=service_context,
# )
# query_engine = index.as_query_engine(
#     similarity_top_k=10,
# )

# final_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=[
#         QueryEngineTool(
#             query_engine=query_engine,
#             metadata=ToolMetadata(
#                 name="Financial statements",
#                 description="Financial information on companies",
#             ),
#         )
#     ],
#     question_gen=question_gen,
#     use_async=True,
# )