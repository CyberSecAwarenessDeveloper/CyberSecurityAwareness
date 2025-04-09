import joblib
import gradio as gr
import torch
import os
import requests
import json
# Download a small model that can run locally
from transformers import AutoModelForCausalLM, AutoTokenizer
# Import the browser-use components
from browser_use import Browser, BrowserConfig
import time
import logging  # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Define available models with descriptions and resource requirements
AVAILABLE_MODELS = {
   
"deepseek-coder-instruct": {
    "description": "DeepSeek Coder V2 Lite (local GGUF via Ollama)",
    "min_ram": "Variable",
    "type": "local",
    "model_path": "C:\\Users\\bbrun\\.lmstudio\\models\\lmstudio-community\\DeepSeek-Coder-V2-Lite-Instruct-GGUF"
}
,
    
    # Recommended API Models (keep as is)
    "openai/gpt-4o-mini": {
        "description": "Affordable OpenAI model with excellent capabilities",
        "min_ram": "N/A - API based",
        "type": "api",
        "provider": "openai",
        "model_id": "gpt-4o-mini"
    },
    "anthropic/claude-3-haiku": {
        "description": "Fast, efficient Anthropic model, lowest cost option",
        "min_ram": "N/A - API based",
        "type": "api",
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307"
    }
}

# Default API keys - LEAVE EMPTY, user should provide their own
API_KEYS = {
    #"openai": "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",
    #"anthropic": "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"
}

# Cost comparison info for API models
API_COST_INFO = {
    "openai/gpt-4o-mini": "~70-80% cheaper than GPT-4o (~$0.0015/query)",
    "anthropic/claude-3-haiku": "~90% cheaper than Claude-3-Opus (~$0.00025/query)",
}

# User-friendly descriptions of models with costs
MODEL_DESCRIPTIONS = {
    "openai/gpt-4o-mini": "✓ RECOMMENDED: Affordable OpenAI model with excellent capabilities (costs ~$0.0015/query)",
    "anthropic/claude-3-haiku": "✓ RECOMMENDED: Fast, efficient Anthropic model, very affordable (costs ~$0.00025/query)",
    #"google/gemma-2b-it": "✓ RECOMMENDED: Best local model for your hardware (free, runs locally)"  # Removed because loading is broken.
}

# Model loading function for local models
def load_local_model(model_name):
    logging.info(f"Loading local model: {model_name}")
    try:
        model_path = AVAILABLE_MODELS[model_name].get("model_path")
        if not model_path:
            raise ValueError(f"Model path not defined for {model_name}")
        # Load model with optimizations for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_path,  # Use the path from AVAILABLE_MODELS
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise

# Function to set API keys
def set_api_key(provider, key):
    API_KEYS[provider] = key
    return f"API key for {provider} has been set"

# Generate response using API models
def generate_with_api(prompt, model_info):
    provider = model_info["provider"]
    model_id = model_info["model_id"]
    
    if API_KEYS.get(provider) == "":
        return f"Please set your {provider} API key first using the API Key tab"
    
    try:
        if provider == "openai":
            return generate_with_openai(prompt, model_id)
        elif provider == "anthropic":
            return generate_with_anthropic(prompt, model_id)
        else:
            return f"Unsupported API provider: {provider}"
    except Exception as e:
        return f"Error generating response with {provider}: {str(e)}"

def generate_with_openai(prompt, model_id):
    headers = {
        "Authorization": f"Bearer {API_KEYS['openai']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def generate_with_anthropic(prompt, model_id):
    headers = {
        "x-api-key": API_KEYS["anthropic"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": model_id,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()["content"][0]["text"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Default model - Updated to use DeepSeek-Coder-V2-Lite-Instruct-GGUF instead of the gated gemma model
default_model_name = "deepseek-coder-instruct"  # make sure this folder exists locally
current_model = None
current_tokenizer = None
current_model_type = "local"

def load_model(model_name):
    global current_model, current_tokenizer, current_model_type
    try:
        model_info = AVAILABLE_MODELS[model_name]
        model_type = model_info["type"]
        current_model_type = model_type
        if model_type == "local":
            model_path = model_info.get("model_path")
            if not os.path.exists(model_path):
                raise ValueError(f"Model path {model_path} does not exist")
            # Skip loading via Transformers; we will use Ollama’s API.
            current_model = None
            current_tokenizer = None
            logging.info(f"Model path exists. Will use Ollama API to generate responses.")
        elif model_type == "api":
            current_model = None
            current_tokenizer = None
        return f"Model {model_name} selected"
    except KeyError as e:
        logging.error(f"Error loading model {model_name}: Model not found. {e}")
        return f"Error: Model {model_name} not found."
    except Exception as ex:
        logging.error(f"An unexpected error occurred loading model {model_name}: {ex}")
        return f"Error: {ex}"


# Initialize default model
try:
    load_model(default_model_name)
except Exception as e:
    logging.error(f"Error loading default model: {str(e)}")
    # Fallback to the most lightweight model - removed gemma as it produced errors
    # default_model_name = "google/gemma-2b-it"  # Removed Gemma fallback since it's likely to error again.  User should select a valid model.
    default_model_name = next(iter(AVAILABLE_MODELS), None) # Fallback to the first available model if the default fails
    if default_model_name:
        try:
            load_model(default_model_name)  # Try to load the new default model.
        except Exception as e2:
            logging.error(f"Error loading fallback model {default_model_name}: {str(e2)}")
            print(f"Error: Could not load any model. Please check model definitions and dependencies.  Original error: {e}")
            exit()  # Exit the program if even the fallback fails.
    else:
         print("Error: No models defined in AVAILABLE_MODELS.")
         exit()


def get_model_insight(user_query):
    insights = {}
    
    try:
        # Awareness model insights
        if any(word in user_query.lower() for word in ["awareness", "training", "knowledge", "understand"]):
            awareness_model = joblib.load('awareness_model.pkl')
            # Convert query to features (simplified for example)
            # In real implementation, you'd need to process the query properly
            insights["awareness"] = "User seems to be asking about security awareness"
        
        # Text threat detection
        if any(word in user_query.lower() for word in ["email", "message", "text", "phishing"]):
            text_vectorizer = joblib.load('text_vectorizer.pkl')
            threat_model = joblib.load('threat_model.pkl')
            text_features = text_vectorizer.transform([user_query])
            prediction = threat_model.predict(text_features)[0]
            insights["threat_detection"] = f"Threat class: {prediction}"
        
        # Vulnerability insights
        if any(word in user_query.lower() for word in ["vulnerability", "cve", "exploit", "patch"]):
            vulnerability_model = joblib.load('vulnerability_model.pkl')
            insights["vulnerability"] = "User is asking about vulnerability management"
        
        # Malware insights
        if any(word in user_query.lower() for word in ["malware", "virus", "ransomware", "trojan"]):
            malware_model = joblib.load('malware_model.pkl')
            insights["malware"] = "User is asking about malware protection"
            
    except Exception as e:
        logging.error(f"Error getting model insights: {str(e)}")
        insights["error"] = str(e)
    
    return insights

from sentence_transformers import SentenceTransformer

# Create embeddings from your cybersecurity datasets
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model

def get_assistant_response(user_input, current_model_name):
    # Gather any model insights if available.
    model_insights = get_model_insight(user_input)
    prompt = f"""
    You are a cybersecurity awareness assistant with expertise in security best practices, threat detection, 
    vulnerability management, and malware protection. Your goal is to educate users about cybersecurity
    in a clear, helpful way.
    
    If available, use these insights from specialized security models:
    {model_insights}
    
    User question: {user_input}
    
    Please provide a helpful, accurate answer about cybersecurity:
    """
    
    try:
        model_info = AVAILABLE_MODELS[current_model_name]
        if model_info["type"] == "local":
            # Use Ollama’s API for local models.
            # Make sure the model name (here: deepseek-coder-instruct) is loaded in your Ollama instance.
            inputs = {"prompt": prompt}
            url = f"http://localhost:11434/models/{current_model_name}/generate"
            response = requests.post(url, json=inputs)
            if response.status_code == 200:
                text_response = response.json().get("response", "").strip()
                return text_response if text_response else "No response generated."
            else:
                return f"Error generating response: {response.text}\nURL used: {url}"
        elif model_info["type"] == "api":
            return generate_with_api(prompt, model_info)
        else:
            return "Model type not recognized. Please select a valid model."
    except Exception as e:
        return f"Error generating response: {str(e)}"

def change_model(new_model_name):
    try:
        result = load_model(new_model_name)
        return f"Successfully switched to model: {new_model_name}"
    except Exception as e:
        return f"Failed to load model {new_model_name}: {str(e)}"

# Browser settings
BROWSER_SETTINGS = {
    
}

# Removed persistent_session parameter from BrowserConfig
def setup_browser_agent():
    try:
        config = BrowserConfig(
            headless=False,              # Show the browser window for live visualization
            screen_recording=True        # Enable visual debugging (if supported)
        )
        return Browser(config)
    except Exception as e:
        logging.error(f"Error setting up browser: {str(e)}")
        return None

# Update search_web() to use the correct browse() method:
def search_web(url, query, current_model_name, history=[]):
    try:
        browser_agent = setup_browser_agent()
        if not browser_agent:
            return "Error: Failed to initialize browser. Please check if browser-use is installed correctly."
        
        # Instead of using navigate, use the built-in browse() method
        logging.info(f"Performing web search for: {query}")
        result = browser_agent.browse(url, query)  # browse() takes (url, query)
        logging.info("Web search completed")
        
        # Build an enhanced prompt using the web page result
        enhanced_prompt = f"""
        You are a cybersecurity awareness assistant with expertise in security best practices, threat detection, 
        vulnerability management, and malware protection. Your goal is to educate users about cybersecurity
        in a clear, helpful way.
        
        I've searched the web for information related to the user's question.
        
        Web search results from {url}:
        {result}
        
        Chat history:
        {history}
        
        User question: {query}
        
        Please provide a helpful answer that combines your knowledge with these web search results:
        """
        
        model_response = get_assistant_response(enhanced_prompt, current_model_name)
        return model_response
        
    except Exception as e:
        error_msg = f"Error during web search: {str(e)}"
        logging.error(error_msg)
        return f"Error occurred during web search. Falling back to model knowledge.\n\n{get_assistant_response(query, current_model_name)}"

# Function to detect if a query needs web search
def needs_web_search(query):
    web_indicators = [
        "search for", "look up", "find information", "latest", "recent", 
        "news", "current", "today", "update", "what is happening",
        "check", "browse", "website", "look online", "online"
    ]
    
    # Check if query contains any web search indicators
    return any(indicator in query.lower() for indicator in web_indicators)

# Modified respond function to include web search when appropriate
def respond(message, history, current_model, use_web=False, web_url=""):
    # Check if web search is requested
    if use_web and web_url:
        logging.info(f"Web search requested for: {message}")
        return search_web(web_url, message, current_model, history)
    
    # Auto-detect if web search might be helpful
    elif use_web and needs_web_search(message):
        logging.info(f"Auto-detected need for web search: {message}")
        # Default to Google for auto-detected searches
        return search_web("https://www.google.com", message, current_model, history)
    
    # Standard response without web search
    else:
        return get_assistant_response(message, current_model)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Cybersecurity Awareness Assistant")
    gr.Markdown("Ask questions about cybersecurity best practices, threats, and protection strategies")
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            # Model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=default_model_name,
                    label="Select AI Model",
                    info="Choose a model based on your hardware capabilities"
                )
                model_info = gr.Markdown(f"Current model: **{default_model_name}**\n\n{AVAILABLE_MODELS[default_model_name]['description']}\n\nMinimum RAM: {AVAILABLE_MODELS[default_model_name]['min_ram']}")
            
            # Show recommended models with cost info
            gr.Markdown("""
            ### Recommended Models:
            - **OpenAI**: gpt-4o-mini - Costs ~70-80% less than GPT-4o (~$0.0015/query)
            - **Anthropic**: claude-3-haiku - Costs ~90% less than Claude-3-Opus (~$0.00025/query)
            """)
            
            # Enable web search checkbox
            use_web_search = gr.Checkbox(
                label="Enable Web Search",
                value=False,
                info="When enabled, the assistant can search the web for timely information"
            )
            
            # Apply model button
            model_button = gr.Button("Apply Model Change")
            model_status = gr.Markdown("Model loaded and ready")
            
            # Chat interface with modified respond function
            chatbot = gr.ChatInterface(
                fn=lambda message, history: respond(
                    message, 
                    history, 
                    model_dropdown.value, 
                    use_web_search.value
                ),
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="Ask about cybersecurity...", container=False, scale=7),
                title="",
            )
        
        with gr.Tab("Web Search"):
            gr.Markdown("## Search the Web for Security Information")
            gr.Markdown("Use this feature to browse specific websites for cybersecurity information")
            
            # URL input
            web_url = gr.Textbox(
                placeholder="https://www.cve.mitre.org",
                label="Website URL",
                value="https://www.google.com"
            )
            
            # Query input
            web_query = gr.Textbox(
                placeholder="Search for latest ransomware threats...",
                label="Search Query"
            )
            
            # Persistent browser toggle
            persistent_browser = gr.Checkbox(
                 label="Keep Browser Open Between Searches",
    value=True,
    info="(Visual only) Not currently used"
            )
            
            # Headless mode toggle
            headless_mode = gr.Checkbox(
                label="Headless Mode (Hide Browser Window)",
                value=False,
                info="When enabled, browser will run in background without visible window"
            )
            
            # Search button
            search_button = gr.Button("Search Web")
            
            # Results display
            search_results = gr.Textbox(
                label="Search Results",
                placeholder="Results will appear here...",
                lines=15
            )
        
        with gr.Tab("API Keys"):
            gr.Markdown("## API Keys for External Models")
            gr.Markdown("Set your API keys to use models from OpenAI and Anthropic. These keys are stored in memory and not saved to disk.")
            
            with gr.Row():
                openai_key = gr.Textbox(
                    placeholder="sk-...",
                    label="OpenAI API Key",
                    type="password"
                )
                openai_button = gr.Button("Set OpenAI Key")
            
            with gr.Row():
                anthropic_key = gr.Textbox(
                    placeholder="sk-ant-...",
                    label="Anthropic API Key",
                    type="password"
                )
                anthropic_button = gr.Button("Set Anthropic Key")
            
            api_status = gr.Markdown("Enter your API keys to use cloud models")
        
        with gr.Tab("Model Info"):
            gr.Markdown("## Available Models")
            
            with gr.Accordion("🖥️ Local Model", open=True):
                gr.Markdown("""
                * DeepSeek-Coder-V2-Lite-Instruct-GGUF - High quality instruction-tuned model for coding and cybersecurity
                """)
            
            with gr.Accordion("☁️ API Models (Requires API Keys)", open=True):
                gr.Markdown("""
                ### Recommended API Models:
                * **openai/gpt-4o-mini** ✓ - Affordable OpenAI model with excellent capabilities (~$0.0015/query)
                * **anthropic/claude-3-haiku** ✓ - Fast, efficient Anthropic model, lowest cost option (~$0.00025/query)
                """)
            
            gr.Markdown("""
            ## Cost Comparison
            
            | Model | Cost per query (approx) | Notes |
            |-------|-------------------------|-------|
            | DeepSeek-Coder-V2-Lite-Instruct-GGUF | Free | Runs locally on your computer (LM Studio required) |
            | GPT-4o-mini | $0.0015 | 70-80% cheaper than full GPT-4o |
            | Claude-3-Haiku | $0.00025 | 90% cheaper than Claude-3-Opus |
            
            *Note: API costs are approximate for typical cybersecurity awareness queries (100-200 words). Actual costs depend on token length.*
            """)
            
            gr.Markdown("""
            ## How to Get More Out of Your Hardware
            
            1. Close other applications to free up RAM before running this app
            2. Restart your computer to maximize available memory
            3. For highest quality responses with minimal hardware requirements, use the API models
            """)
    
    # Handle model change
    def update_model_info(model_name):
        info = f"Current model: **{model_name}**\n\n"
        
        # Add custom description if available
        if model_name in MODEL_DESCRIPTIONS:
            info += f"{MODEL_DESCRIPTIONS[model_name]}\n\n"
        else:
            info += f"{AVAILABLE_MODELS[model_name]['description']}\n\n"
        
        # Add RAM requirements for local models or cost info for API models
        if AVAILABLE_MODELS[model_name]['type'] == 'local':
            info += f"Minimum RAM: {AVAILABLE_MODELS[model_name]['min_ram']}"
        else:
            if model_name in API_COST_INFO:
                info += f"Cost: {API_COST_INFO[model_name]}"
        
        return info
    
    # Update browser settings
    def update_browser_settings(persistent, headless):
        # You can store these settings for UI purposes but do not pass them to BrowserConfig.
        BROWSER_SETTINGS["headless"] = headless
        return f"Browser settings updated: Headless={headless}"
    
    # Function to perform web search
    def perform_web_search(url, query, model_name):
        if not url or not query:
            return "Please provide both a URL and search query"
        
        return search_web(url, query, model_name)
    
    # Connect API key buttons
    openai_button.click(set_api_key, inputs=[gr.Textbox(value="openai", visible=False), openai_key], outputs=api_status)
    anthropic_button.click(set_api_key, inputs=[gr.Textbox(value="anthropic", visible=False), anthropic_key], outputs=api_status)
    
    # Connect model dropdown
    model_dropdown.change(update_model_info, model_dropdown, model_info)
    model_button.click(change_model, model_dropdown, model_status)
    
    # Connect browser settings
    persistent_browser.change(
        lambda p, h: update_browser_settings(p, h),
        inputs=[persistent_browser, headless_mode],
        outputs=gr.Textbox(visible=False)
    )
    
    headless_mode.change(
        lambda p, h: update_browser_settings(p, h),
        inputs=[persistent_browser, headless_mode],
        outputs=gr.Textbox(visible=False)
    )
    
    # Connect web search button
    search_button.click(
        perform_web_search,
        inputs=[web_url, web_query, model_dropdown],
        outputs=search_results
    )

demo.launch()