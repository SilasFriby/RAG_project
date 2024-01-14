import os
import json
import openai
from dotenv import load_dotenv
from llama_index.llms import Ollama, OpenAI #, Perplexity 
from custom_classes.custom_perplexity_llm import CustomPerplexityLLM
from llama_index.llms import ChatMessage
import ast

# Initialize variables
documents_file_path = "data/test.jsonl" #statements_id_title_text_sub.jsonl"
llm_model_names = ["llama2", "gpt-3.5-turbo-0613", "mixtral-8x7b-instruct"]#"mistral-7b-instruct"]
llm_temp = 0
llm_response_max_tokens = 1024
choose_llm_model = 2

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

# LLM
if choose_llm_model == 0:
    llm_model= Ollama(
        model=llm_model_names[choose_llm_model], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens
    )
elif choose_llm_model == 1:
    llm_model = OpenAI(
        model=llm_model_names[choose_llm_model], 
        temperature=llm_temp, 
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("OPENAI_API_KEY")
    )
elif choose_llm_model == 2:
    llm_model = CustomPerplexityLLM(
        model=llm_model_names[choose_llm_model],
        temperature=llm_temp,
        max_tokens=llm_response_max_tokens,
        api_key=os.getenv("PERPLEXITY_API_KEY"), 
    )

# Read documents 
documents_info = []
with open(documents_file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        documents_info.append(data)

# Loop over titles and generate prompts
titles = [doc["title"] for doc in documents_info]
len(titles)

# Prompt to extract time period
PROMPT_PREFIX = """
Act as an expert in named entity regognition. \n \
Your job is to extract the entity "time period" from the title of financial statements. \n \

Below you see Example 1 which shows you how the output list should look given an input list. \
If no time period is present in the title, the output list should be an empty list for that entry, see the second entry in Example 1. \n\n \
Your job is to complete the output list in example 2. Hence, your answer should be in the following structure: \
\n\n \
[time period 1, \n \ 
time period 2, \n \
time period 3, \n \
...] \n\n \
where each time period is a list of the form: [X, Y, Z] where X is the year (e.g 2023) and Y and Z are months, hence they have to be integers >= 1 and <=12 with Y being less than or equal to Z. \n\n \

Remember that if the time period consists of a whole year, then Y and Z should be 1 and 12 respectively. \n\n \

Provide nothing else than the output list. Note that the output list should have exactly the same length as the input list. \n\n \

Example 1: \n\n \

Input list: \n \
["Xbrane Biopharma releases interim report for January – June 2023", \n \
"First Tin PLC  - Final results from Tin Beetle drilling at Taronga", \n \
"Q2 2023: Continued organic growth and further strengthened market position in the Nordics", \n \\
"Interim Financial Report 2023: Cadeler presents positive results and an outlook exceeding expectations", \n \
"KAP AG: DEVELOPMENT IN THE FIRST HALF OF 2023 IN LINE WITH EXPECTATIONS", \n \
"Top Strike Announces 2023 Fourth Quarter and Annual Financial Results ending April 30, 2023 and Corporate Update"] \n\n \

Output list: \n \
[[2023, 1, 6], \n \
[], \n \
[2023, 4, 6], \n \
[2023, 1, 12] \n \
[2023, 1, 6] \n \
[2023, 1, 12]] \n\n
"""

PROMPT_SUFFIX = f"""
Example 2: \n\n \

Input list: \n \
{titles} \n\n \

Output list: \n \
"""

PROMPT = PROMPT_PREFIX + PROMPT_SUFFIX

# Message to the LLM
messages_dict = [
    {"role": "user", "content": PROMPT},
]
messages = [ChatMessage(**msg) for msg in messages_dict]

# Generate response from the LLM
response = llm_model.chat(messages)
print(response)
response_dict = response.dict()["message"]["content"]
response_dict = response_dict.replace('[[', '[')
response_dict = response_dict.replace(']]', ']')
time_periods = response_dict.split(", \n")
len(time_periods)
time_periods = [ast.literal_eval(time_period) for time_period in time_periods]
len(time_periods)

# Add time period to documents
for i, doc in enumerate(documents_info):
    doc["time_period_2"] = time_periods[i]

# Save documents
with open(documents_file_path, "w") as file:
    for doc in documents_info:
        json.dump(doc, file)
        file.write("\n")


# remove "time_period" and "time_period_2" from documents and save to file
for doc in documents_info:
    del doc["time_period"]

with open("data/test.jsonl", "w") as file:
    for doc in documents_info:
        json.dump(doc, file)
        file.write("\n")


    


