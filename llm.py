import os

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv(find_dotenv())

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_HUGGINGFACE_MODEL = "google/flan-t5-xxl"

PROMPT = PromptTemplate.from_template(
    """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""
)


def get_prompt(input_text):
    return PROMPT.format(input_text=input_text)


def get_provider():
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
    if provider in {"huggingface", "hf"}:
        return "huggingface"

    return "langchain"


def build_chain(model=None, provider=None, llm=None):
    selected_llm = llm or make_llm(provider=provider, model=model)
    return PROMPT | selected_llm | StrOutputParser()


def make_llm(provider=None, model=None):
    provider = (provider or get_provider()).strip().lower()

    if provider in {"huggingface", "hf"}:
        from langchain_community.llms import HuggingFaceHub

        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        return HuggingFaceHub(
            repo_id=model or os.getenv("HUGGINGFACEHUB_REPO_ID", DEFAULT_HUGGINGFACE_MODEL),
            task=os.getenv("HUGGINGFACEHUB_TASK", "text2text-generation"),
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": 0, "max_new_tokens": 512},
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_completion(input_text, model=None, provider=None):
    chain = build_chain(model=model, provider=provider)
    return chain.invoke({"input_text": input_text})
