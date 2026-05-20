from dotenv import find_dotenv, load_dotenv

import llm

load_dotenv(find_dotenv())


def get_completion(input_text, model=None):
    return llm.get_completion(input_text, model=model)


def get_prompt(input_text):
    return llm.get_prompt(input_text)
