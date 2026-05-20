import llm


def get_completion(input_text, model=None):
    return llm.get_completion(input_text, model=model)


def get_prompt(input_text):
    return llm.get_prompt(input_text)
