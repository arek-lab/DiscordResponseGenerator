import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-nano")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")

def get_openai():
    return ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY)
def get_anthropic():
    return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
