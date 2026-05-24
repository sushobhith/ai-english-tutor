import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())

DEFAULT_PROVIDER = "langchain"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_HUGGINGFACE_REPO_ID = "google/flan-t5-large"

PROMPT_TEMPLATE = """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use,
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def get_prompt(input_text):
    return PROMPT_TEMPLATE.format(input_text=input_text)


def _normalize_provider(provider):
    provider_name = (provider or DEFAULT_PROVIDER).strip().lower()
    if provider_name not in {"langchain", "huggingface"}:
        return DEFAULT_PROVIDER
    return provider_name


def _sync_legacy_huggingface_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


def build_chain(model=None, llm=None):
    model_name = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    chat_model = llm or ChatOpenAI(model=model_name, temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | chat_model | StrOutputParser()


def build_huggingface_chain(llm=None):
    _sync_legacy_huggingface_token()
    huggingface_model = llm or HuggingFaceHub(
        repo_id=os.getenv("HUGGINGFACEHUB_REPO_ID") or DEFAULT_HUGGINGFACE_REPO_ID,
        model_kwargs={"temperature": 0, "max_new_tokens": 1024},
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | huggingface_model | StrOutputParser()


def get_completion(text, model=None):
    provider = _normalize_provider(os.getenv("LLM_PROVIDER"))
    if provider == "huggingface":
        return build_huggingface_chain().invoke({"input_text": text})

    return build_chain(model=model).invoke({"input_text": text})
