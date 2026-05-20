import llm

repo_id = 'gpt-3.5-turbo'


def get_completion(input_text, model=repo_id):
    return llm.get_completion(input_text, model=model)


def get_prompt(input_text):
    return llm.get_prompt(input_text)
