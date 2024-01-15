"""Autoretriever prompts."""


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

# NOTE: these prompts are inspired from langchain's self-query prompt,
# and adapted to our use case.
# https://github.com/hwchase17/langchain/tree/main/langchain/chains/query_constructor/prompt.py


PREFIX = """ \n \
Your job is to create a structured request for a database search. \n\n 

More specifically, given a query and metadata your job is to structure the query in such a way that it matches the request schema provided below. \n\n \

<<Structured Request Schema>>: \n \
Your output must be a markdown code snippet with a JSON object formatted in the following schema \n\n \

{schema_str} \n\n \

In cases where no filters should be used please return [] for the filter value. \n \
Make sure that filters only refer to variables that exist in the metadata. \n \
The query itself should remain unchanged in the structured request, see Example 1. \n\n \

Your job is to complete the structured request in Example 2. \n \
Provide nothing else but the structured request. \n\n \



"""
example_info = VectorStoreInfo(
    content_info="Classic literature",
    metadata_info=[
        MetadataInfo(name="author", type="str", description="Author name"),
        MetadataInfo(
            name="book_title",
            type="str",
            description="Book title",
        ),
        MetadataInfo(
            name="year",
            type="int",
            description="Year Published",
        ),
        MetadataInfo(
            name="pages",
            type="int",
            description="Number of pages",
        ),
        MetadataInfo(
            name="summary",
            type="str",
            description="A short summary of the book",
        ),
    ],
)

example_query = "What are some books by Jane Austen published after 1813 that explore the theme of marriage for social standing?"

example_output = VectorStoreQuerySpec(
    query=example_query,
    filters=[
        MetadataFilter(key="year", value="1813", operator=FilterOperator.GT),
        MetadataFilter(key="author", value="Jane Austen"),
    ],
)

EXAMPLES = f""" \

Example 1 \n\n \

Metadata: \n \
```json \n \
{example_info.json(indent=4)} \n \
``` \n\n \

User Query: \n \
{example_query} \n\n \

Structured Request: \n \
```json \n \
{example_output.json()} \n\n \
```
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)


SUFFIX = """ \n \
Example 2 \n\n \

Metadata: \n \
```json \n \
{info_str} \n \
``` \n\n \

User Query: \n \
{query_str} \n\n \

Structured Request: \n \
"""

CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX


# deprecated, kept for backwards compatibility
"""Vector store query prompt."""
VectorStoreQueryPrompt = PromptTemplate

CUSTOM_VECTOR_STORE_QUERY_PROMPT = PromptTemplate(
    template=CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL,
    prompt_type=PromptType.VECTOR_STORE_QUERY,
)

# print(CUSTOM_VECTOR_STORE_QUERY_PROMPT_TMPL)
# The query string should contain only text that is expected to match the contents of documents. \n \
# Any conditions in the filter should not be mentioned in the query as well. \n \
# The query string should contain only text that is expected to match the contents of documents. \n \
# Any conditions in the filter should not be mentioned in the query as well. \n \
