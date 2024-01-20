from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

CUSTOM_CHOICE_SELECT_PROMPT = """ \n \
    Given a user question, and a list of documents, please output a list of of numbers corresponding to the documents \n \
    you should consult in order to answer the user question. \n \
    Each document has a number next to it along with some metadata (e.g. file path, title, etc.) followed by the document text. \n\n \
    
    Your job is to complete the output in example 2. \n\n \
    
    Below you see Example 1, which shows you an example of how output should look given a user question and a document list. \n \
    Note that the relevance score for document 1 is 10, since it holds the needed information to answer the question. \
    The relevance score for document 2 is 7, because it elaborates on the results for Apple. \
    Document 10 is deemed irrelevant and therefore not included. \n\n \
     
    
    Example 1: \n\n \
    Question: What was the profit of Apple in 2022? \n \
    Document 1: \n \
    file_name: apple_results_2022.txt \n \
    Title: Apple results for 2022 \n \
    Apple Inc. announced its financial results for the year ended 2022. The company reported a Total Revenue of $350 billion, demonstrating a robust market performance. The Cost of Goods Sold (COGS) was recorded at $220 billion, primarily driven by manufacturing costs. This resulted in a Gross Profit of $130 billion, reflecting Apple's effective cost management and premium product pricing. \n \
    Document 2: \n \
    file_name: apple_extras_2022.txt \n \
    Title: Additional information on Apple results for 2022 \n \
    In 2022, Apple Inc. experienced significant growth in international markets, with international sales accounting for approximately 60% of its total revenue. This demonstrates the company's strong global presence and its ability to appeal to diverse markets. The year was marked by the successful launch of several key products, including a new iPhone model and an updated MacBook series, which significantly contributed to the revenue boost. \n \
    ...\n \
    Document 10:\n \
    file_name: microsoft_results_2022.txt \n \
    Title: Microsoft results for 2022 \n \ \n \
    In 2022, Microsoft reported a significant increase in cloud computing revenues, bolstering its overall financial performance for the year. \n\n \
    
    Output: \n \
    Doc: 1, Relevance: 10\n \
    Doc: 2, Relevance: 7\n \
    
    Example 2: \n\n \
    Question: {query_str} \n \
    {context_str} \n\n \
    Output: \n\n \
    
    IMPORTANT: \n \
    Your output should not contain an explanation for your choice of documents and relevance scores, only the list. \
    
"""

# print(CUSTOM_CHOICE_SELECT_PROMPT)

CUSTOM_CHOICE_SELECT_PROMPT = PromptTemplate(
    CUSTOM_CHOICE_SELECT_PROMPT, prompt_type=PromptType.CHOICE_SELECT
)

    # IMPORTANT: \n \
    # Provide nothing else but the output from Example 2, i.e a list in the format 
    # \n\n \
    # Doc: X, Relevance: A \n \
    # Doc: Y, Relevance: B \n \
    # \n\n \
    # where X and Y are integers corresponding to the documents you should consult in order to answer the user question. \n \
    # And A and B are relevance scores from 1 to 10 based on how relevant you think the document is to the user question. \n \
    # 10 is given if the document is very relevant and 1 is given if the document is not relevant at all. \n \
    # Your output should not contain an explanation for your choice of documents and relevance scores, only the list. \

