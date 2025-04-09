import os
import gradio as gr
import logging
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BROWSER_AGENT_API = "http://localhost:8080/api/agent"
OLLAMA_MODEL_NAME = "deepseek-coder-v2"
OLLAMA_ENDPOINT = "http://localhost:11434"

# Save search history
def save_search_history(query, result):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("search_history.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Query: {query}\nResult: {result}\n\n")

# Call browser-use API
def query_browser_agent(message):
    try:
        response = requests.post(
            BROWSER_AGENT_API,
            json={"input": message}
        )
        result = response.json()
        return result.get("output", "No output from browser agent.")
    except Exception as e:
        logger.error(f"Browser agent API error: {e}")
        return None

# Call Ollama LLM
def query_llm(message):
    try:
        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": message}
        )
        result = response.json()
        return result.get("response", "No output from LLM.")
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None

# Chat function
def chat(message, history):
    logger.info(f"Received input: {message}")
    try:
        response = query_browser_agent(message)
        if not response:
            logger.warning("Browser agent failed. Falling back to LLM.")
            response = query_llm(message)
        if not response:
            response = "All agents failed."
        logger.info(f"Response: {response}")
        save_search_history(message, response)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred: {e}"

# Gradio interface
iface = gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(height=500, type="messages"),
    textbox=gr.Textbox(placeholder="Ask me anything about cybersecurity or search the web...", container=False, scale=7),
    title="Cyber Security AI Assistant",
    description="Ask questions, search the web, and learn using AI tools.",
    theme="soft",
    examples=["What is phishing?", "Search for AI in cybersecurity.", "What are some common threats in 2025?"]
)

if __name__ == "__main__":
    iface.launch(share=False)
