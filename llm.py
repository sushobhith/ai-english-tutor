import os

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    pass


DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_HUGGINGFACE_REPO_ID = "google/flan-t5-large"

ANALYSIS_PROMPT_TEMPLATE = """proof read this text: ```{input_text}```

Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use,
Comprehension and Responsiveness, Content and Structure and Creativity and Originality.

For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
"""


def get_prompt(input_text):
    return ANALYSIS_PROMPT_TEMPLATE.format(input_text=input_text)


def get_completion(input_text):
    if _selected_provider() == "huggingface":
        return get_huggingface_completion(input_text)

    return get_langchain_completion(input_text)


def get_langchain_completion(input_text):
    response = build_chain().invoke({"input_text": input_text})
    return _message_content(response)


def get_huggingface_completion(input_text):
    _sync_huggingface_token()
    response = build_huggingface_chain().invoke({"input_text": input_text})
    return _message_content(response)


def build_chain(llm=None):
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
    model = llm or ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
    )
    return prompt | model


def build_huggingface_chain(llm=None):
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import HuggingFaceHub

    prompt = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
    model = llm or HuggingFaceHub(
        repo_id=os.getenv("HUGGINGFACE_REPO_ID", DEFAULT_HUGGINGFACE_REPO_ID),
        model_kwargs={"temperature": 0, "max_length": 1024},
    )
    return prompt | model


def _selected_provider():
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
    if provider == "huggingface":
        return "huggingface"

    return "langchain"


def _sync_huggingface_token():
    legacy_token = os.getenv("HUGGINGFACE_API_KEY")
    if legacy_token and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = legacy_token


def _message_content(response):
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if content is not None:
        return content

    return str(response)
