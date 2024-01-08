import json
from typing import Sequence
from llama_index.prompts.base import PromptTemplate
from llama_index.question_gen.types import SubQuestion
from llama_index.tools.types import ToolMetadata

# deprecated, kept for backward compatibility
SubQuestionPrompt = PromptTemplate

def build_tools_text(tools: Sequence[ToolMetadata]) -> str:
    tools_dict = {}
    for tool in tools:
        tools_dict[tool.name] = tool.description
    return json.dumps(tools_dict, indent=4)
  
PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json format that when composed can help answer the full user question. \
Below you see Example 1, which shows you how output should look given an input. \
Your job is to complete the output in example 2. Hence, your answer should in the folowwing structure: \
\n\n \
```json  \n\
X \
\n ``` \
\n\n \
where X is a json dictionary containing sub-questions. Provide nothing else but the sub-questions in the above format. \
"""

example_query_str = (
    "Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021"
)
example_tools = [
    ToolMetadata(
        name="Financial statements",
        description="Financial information on companies",
    ),
]
example_tools_str = build_tools_text(example_tools)
example_output = [
    SubQuestion(
        sub_question="What is the revenue growth of Uber", tool_name="Financial statements"
    ),
    SubQuestion(sub_question="What is the EBITDA of Uber", tool_name="Financial statements"),
    SubQuestion(
        sub_question="What is the revenue growth of Lyft", tool_name="Financial statements"
    ),
    SubQuestion(sub_question="What is the EBITDA of Lyft", tool_name="Financial statements"),
]
example_output_str = json.dumps([x.dict() for x in example_output], indent=4)
example_output_str = json.dumps([x.dict() for x in example_output], indent=4)


EXAMPLES = f"""\
\n\n # Example 1 \n\n
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

SUFFIX = """\
\n\n # Example 2 \n\n
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX
