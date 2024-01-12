CUSTOM_QUESTION_GEN_TMPL = """\
Here is the context:
{context_str}

Given the contextual information above, your job is to generate {num_questions} questions that the context can provide answers to. \n \
Do not create long and complicated questions, but attempt to keep the questions simple and clear \

"""