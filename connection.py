from agents import OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemnini_api_key = os.getenv("GEMINI_API_KEY")

if not gemnini_api_key:
    raise ValueError("GEMINI API key not found")

external_client = AsyncOpenAI(
  api_key=gemnini_api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)


config = RunConfig(
    model=model,
    model_provider=external_client
)