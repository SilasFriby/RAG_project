from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

CHOICE_SELECT_PROMPT_TMPL = """ \n \
    Given a user question, and a list of documents, please output a list of of numbers corresponding to the documents \n \
    you should consult in order to answer the user question. \n \
    Each document has a number next to it along with some metadata (e.g. file path, title, etc.) followed by the document text. \n \
    Below you see Example 1, which shows you how output should look given a user question and a document list. \n \
    Your job is to complete the output in example 2. Hence, your answer should be in the following structure: 
    \n\n \
    Doc: X, Relevance: k\n \
    Doc: Y, Relevance: j\n \
    Doc: Z, Relevance: i \
    \n\n \
    where X, Y, Z are numbers corresponding to the documents you should consult in order to answer the user question. And k, j, i are relevance scores from 1-10 based on \
    how relevant you think the document is to the question.\n \
    Provide nothing else but the list of numbers in the above format.\n\n \
    
    Example 1: \n\n \
    Question: ..some question.. \n \
    Document 1: \n \
    metadata \n \
    text \n \
    Document 2: \n \
    metadata \n \
    text \n \
    ...\n \
    Document 10:\n \
    metadata \n \
    text \n\n \
    
    Output: \n \
    Doc: 9, Relevance: 7\n \
    Doc: 3, Relevance: 4\n \
    Doc: 7, Relevance: 3\n\n \
    
    Example 2: \n\n \
    Question: {query_str} \n \
    {context_str} \n\n \
    Output: \n \
"""

CHOICE_SELECT_PROMPT = PromptTemplate(
    CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)

