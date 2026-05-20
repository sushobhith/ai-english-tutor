import os
from llm import create_llm_provider

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:
    def find_dotenv():
        return ""

    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv(find_dotenv())

if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def get_completion(input_text, model=DEFAULT_MODEL, provider=None):
    llm_provider = create_llm_provider(model=model, provider=provider)
    return llm_provider.complete(get_prompt(input_text))

def get_prompt(input_text):
  prompt = f"""proof read this text: ```{input_text}``` 

  Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
  Comprehension and Responsiveness, Content and Structure and Creativity and Originality. \

  For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
  """
  return prompt
