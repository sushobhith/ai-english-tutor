import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_HUGGINGFACE_REPO_ID = "google/flan-t5-large"


def get_prompt(input_text):
    return f"""proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def build_chain(llm=None):
    prompt = PromptTemplate.from_template("{analysis_prompt}")
    return prompt | (llm or _create_chat_openai())


def get_completion(input_text):
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
    if provider == "huggingface":
        return _get_completion_huggingface(input_text)
    return _get_completion_langchain(input_text)


def _get_completion_langchain(input_text):
    return _run_chain(_create_chat_openai(), input_text)


def _get_completion_huggingface(input_text):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

    from langchain_community.llms import HuggingFaceHub

    model = HuggingFaceHub(
        repo_id=os.getenv("HUGGINGFACEHUB_REPO_ID", DEFAULT_HUGGINGFACE_REPO_ID),
        model_kwargs={"temperature": 0},
    )
    return _run_chain(model, input_text)


def _create_chat_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _run_chain(model, input_text):
    result = build_chain(model).invoke({"analysis_prompt": get_prompt(input_text)})
    return _stringify_generation(result)


def _stringify_generation(result):
    if hasattr(result, "content"):
        return result.content
    return str(result)
