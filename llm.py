import os
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_HUGGINGFACE_REPO_ID = "google/flan-t5-large"

PROMPT_TEMPLATE = """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use,
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def get_prompt(input_text: str) -> str:
    return PROMPT_TEMPLATE.format(input_text=input_text)


def build_chain(llm: Any | None = None):
    model = llm or ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=0,
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | model | StrOutputParser()


def get_completion(input_text: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()

    if provider == "huggingface":
        return get_huggingface_completion(input_text)

    return get_langchain_completion(input_text)


def get_langchain_completion(input_text: str) -> str:
    return build_chain().invoke({"input_text": input_text})


def get_huggingface_completion(input_text: str) -> str:
    from langchain_community.llms import HuggingFaceHub

    model = HuggingFaceHub(
        repo_id=os.getenv("HUGGINGFACE_REPO_ID", DEFAULT_HUGGINGFACE_REPO_ID),
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"),
        model_kwargs={"temperature": 0, "max_length": 1024},
    )
    result = model.invoke(get_prompt(input_text))
    return result if isinstance(result, str) else str(result)
