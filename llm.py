from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
import os

load_dotenv(find_dotenv())

DEFAULT_OPENAI_MODEL = 'gpt-3.5-turbo'
DEFAULT_HUGGINGFACE_REPO = 'google/flan-t5-xxl'

PROMPT_TEMPLATE = """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def build_prompt_text(input_text):
    return PROMPT_TEMPLATE.format(input_text=input_text)


def build_prompt_template():
    return PromptTemplate.from_template(PROMPT_TEMPLATE)


def get_provider():
    provider = os.getenv('LLM_PROVIDER', 'langchain').strip().lower()
    if provider not in ('langchain', 'huggingface'):
        return 'langchain'
    return provider


def build_chain(llm_instance=None, model=None):
    if llm_instance is None:
        llm_instance = build_llm(model=model)
    return build_prompt_template() | llm_instance


def build_llm(model=None):
    if get_provider() == 'huggingface':
        return build_huggingface_llm()
    return build_langchain_llm(model=model)


def build_langchain_llm(model=None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL,
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
    )


def build_huggingface_llm():
    from langchain_community.llms import HuggingFaceHub

    token = os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
    return HuggingFaceHub(
        repo_id=os.getenv('HUGGINGFACEHUB_REPO_ID') or DEFAULT_HUGGINGFACE_REPO,
        huggingfacehub_api_token=token,
        model_kwargs={'temperature': 0, 'max_new_tokens': 512},
    )


def get_completion(input_text, model=None):
    result = build_chain(model=model).invoke({'input_text': input_text})
    if hasattr(result, 'content'):
        return result.content
    return str(result)
