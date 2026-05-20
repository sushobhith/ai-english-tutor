from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-3.5-turbo"

PROMPT_TEMPLATE = """proof read this text: ```{input_text}``` 

  Summarize the analysis on the basis of Grammar and Syntax, Vocabulary and Language Use, \
  Comprehension and Responsiveness, Content and Structure and Creativity and Originality. \

  For each parameter respond in a JSON structure, give marks out of 10 and suggestion on how to improve it.
  """


def get_prompt(input_text):
    return PROMPT_TEMPLATE.format(input_text=input_text)


def build_chain(llm=None, prompt_template=None):
    """Build a LangChain runnable without making network calls unless no LLM is injected."""
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(prompt_template or PROMPT_TEMPLATE)
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            temperature=0,
        )
    return prompt | llm


def _response_to_text(response):
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _get_langchain_completion(input_text):
    chain = build_chain()
    return _response_to_text(chain.invoke({"input_text": input_text}))


def _get_legacy_completion(input_text, model=DEFAULT_MODEL):
    from openai import OpenAI

    huggingface_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if huggingface_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    client = OpenAI()
    messages = [{"role": "user", "content": get_prompt(input_text)}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def get_completion(input_text):
    provider = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
    if provider == "huggingface":
        return _get_legacy_completion(input_text)
    return _get_langchain_completion(input_text)

