"""Autoretriever prompts."""


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
    FilterCondition,
    MetadataFilters,
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

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value. \n\n \

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k. \n\n \

Your job is to complete the structured request in Example 2. \n \
Provide nothing else but the structured request. \n\n \



"""
example_info = VectorStoreInfo(
    content_info="Classic literature",
    metadata_info=[
        MetadataInfo(
            name="company_name",
            type="str",
            description=(
                "The name of the company that published the financial statement, e.g. Apple"
            ),
        ),
    ],
)

example_query = "What were the driving financial metrics for Apple in 2022?"

example_output = VectorStoreQuerySpec(
    query="What were the driving financial metrics in 2022?",
    filters=[
        MetadataFilter(key="company_name", value="Apple", operator=FilterOperator.EQ),
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
