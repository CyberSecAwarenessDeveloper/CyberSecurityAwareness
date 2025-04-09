import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BROWSER_AGENT_URL = os.getenv("BROWSER_AGENT_URL", "http://localhost:7788/api/agent")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-coder-v2")


def query_browser_agent(message):
    try:
        response = requests.post(
            BROWSER_AGENT_URL,
            json={"input": message}
        )
        result = response.json()
        return result.get("output", "No output from browser agent.")
    except Exception as e:
        logger.error(f"Browser agent API error: {e}")
        return None


def query_llm(message):
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": message}
        )
        result = response.json()
        return result.get("response", "No output from LLM.")
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None
