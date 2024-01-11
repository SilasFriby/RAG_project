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
Your goal is to structure the user's query given metadata. \n \
The user query should be structured in such a way that it matches the request schema provided below. \n\n \

Structured Request Schema: \n \
When responding use a markdown code snippet with a JSON object formatted in the following schema \n\n \

{schema_str} \n\n \

The query string should contain only text that is expected to match the contents of documents. \
Any conditions in the filter should not be mentioned in the query as well. \
Also make sure that filters only refer to attributes that exist in the metadata. \n\n \

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k. \n\n \

Below you see Example 1, which shows you how the structured request should look given a metadata filter. \n\n \

Your job is to complete the structured request in example 2 given the user query, the metadata filter and the additional information. \
Provide nothing else but the structured request in the format specified above. \n\n \

At last, but not least, assume that the descriptions in the metadata are written by an expert, therefore your must pay close attention to the descriptionså. \n \
Hence, it is important that filters are only applied when and if text from the user query matches the decriptions from metadata. \n \
In cases where no filters should be used please return [] for the filter value. \n\n \

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
    query="Books related to theme of marriage for social standing",
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
