# src/web/cyber_webui.py
import logging
import requests
import os
import sys
from dotenv import load_dotenv
import gradio as gr

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from src.ml.predictor import predict_by_category

# Load .env variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
fastapi_app = FastAPI()

# Allow local CORS for frontend integration
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Python path to find webUIFiles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the web-ui interface
from src.webUIFiles import webui as web_ui_app

# Home route
@fastapi_app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Cyber Awareness</title></head>
        <body>
            <h2>Welcome to the Cyber Security Dashboard</h2>
            <ul>
                <li><a href="/chat">Go to Cyber Chatbot</a></li>
                <li><a href="/assistant">Open web-ui Assistant</a></li>
            </ul>
        </body>
    </html>
    """

# Query Ollama
OLLAMA_URL = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "deepseek-coder-v2:latest")

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "No response from Ollama.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama HTTP error: {e}")
        return f"Ollama HTTP error: {e}"

# Chat prediction logic
def predict(message: str, history: list = None):
    logger.info(f"Received input: {message}")

    message_lower = message.lower()

    # Use local model if specific keywords are detected
    if "phishing" in message_lower:
        result = predict_by_category(message, "awareness")
        return f"🛡️ Awareness model says: {result}"
    elif "malware" in message_lower:
        result = predict_by_category(message, "malware")
        return f"🦠 Malware model says: {result}"
    elif "threat" in message_lower:
        result = predict_by_category(message, "threat")
        return f"🚨 Threat model says: {result}"
    elif "vulnerability" in message_lower:
        result = predict_by_category(message, "vulnerability")
        return f"🔍 Vulnerability model says: {result}"

    # Use browser agent if mentioned
    if "agent" in message_lower or "check" in message_lower:
        try:
            agent_response = requests.post(
                "http://127.0.0.1:7788/api/agent",
                json={"input": message},
                timeout=60
            )
            if agent_response.status_code == 200:
                return agent_response.json().get("output", "Agent responded but no output.")
            return f"Agent call failed: {agent_response.status_code} - {agent_response.text}"
        except Exception as e:
            return f"Agent error: {e}"

    # Fallback to general-purpose LLM (Ollama)
    return query_ollama(prompt=message)


# Gradio ChatInterface
gradio_chat = gr.ChatInterface(
    fn=predict,
    chatbot=gr.Chatbot(height=500, type="messages"),
    textbox=gr.Textbox(
        placeholder="Ask cybersecurity questions or interact with the browser agent...",
        container=False,
        scale=7
    ),
    title="Cyber Security Awareness Assistant",
    description="Ask questions, analyze browser content, and learn about cyber threats.",
    theme="soft",
    examples=[["Is this email safe?"], ["Run awareness test"], ["Check if this link is phishing"]]
)

# Mount routes
fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_chat, path="/chat")
fastapi_app = gr.mount_gradio_app(fastapi_app, web_ui_app.create_ui(theme_name="Ocean"), path="/assistant")

# Entry point
app = fastapi_app
