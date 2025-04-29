# src/web/cyber_webui.py
import gradio as gr, logging, time
from fastapi import FastAPI
from src.ml.predictor import answer_with_rag, check_ollama_running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define available roles for the assistant
ROLES = {
    "Security Expert": "You are a cybersecurity expert with deep technical knowledge.",
    "Security Educator": "You are a cybersecurity educator, focused on teaching concepts clearly.",
    "Security Analyst": "You are a security analyst who investigates threats and vulnerabilities.",
    "Security Advisor": "You are a security advisor who provides practical recommendations."
}

def handle_prompt(user_message, history, role):
    """Process the user message and return updated history"""
    logger.info(f"Q: {user_message} (Role: {role})")
    
    # Initialize empty history if None
    if history is None:
        history = []
    
    # Add user message to history immediately
    history.append({"role": "user", "content": user_message})
    
    # Create a placeholder for the assistant's response
    history.append({"role": "assistant", "content": "Thinking..."})
    
    # Return the history with the placeholder, and a function that will update it
    return history, user_message, role

def stream_response(history, user_message, role):
    """Generate a streaming response and update the history"""
    # Track time for performance metrics
    start_time = time.time()
    
    # The current state of the streaming response
    response_so_far = ""
    
    # Define callback for streaming updates
    def update_response(content):
        nonlocal response_so_far
        response_so_far = content
        
        # Update the last message in history (the assistant's response)
        history[-1]["content"] = content
        return history
    
    try:
        # Generate response with streaming updates
        answer = answer_with_rag(user_message, k=3, role=role, stream_callback=update_response)
        
        # Ensure the final response is properly set
        if not response_so_far:
            history[-1]["content"] = answer
            
        # Log timing information
        end_time = time.time()
        logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        history[-1]["content"] = "I encountered an error while processing your request."
    
    return history

with gr.Blocks(title="Cybersecurity Awareness Assistant") as interface:
    # Status indicator
    with gr.Row():
        status_indicator = gr.Markdown(
            "🟢 Ollama server is running" if check_ollama_running() else "🔴 Ollama server is offline - Using fallback mode"
        )
    
    gr.Markdown("# Cybersecurity Awareness Assistant")
    
    # Role selector
    role_dropdown = gr.Dropdown(
        choices=list(ROLES.keys()),
        value="Security Expert",
        label="Select Assistant Role",
        info="Choose how the assistant should respond to your questions"
    )
    
    # Chat interface with messages format
    chatbot = gr.Chatbot(
        value=[], 
        label="Cybersecurity Chat",
        height=500,
        type="messages",  # Important for proper formatting
        show_copy_button=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Ask anything about cybersecurity",
            placeholder="e.g. How does a DDoS attack work?",
            scale=9
        )
        submit_btn = gr.Button("Submit", scale=1, variant="primary")
    
    # Clear button
    clear_btn = gr.ClearButton([msg, chatbot], value="Clear Chat")
    
    # Performance metrics
    with gr.Accordion("Performance Metrics", open=False):
        with gr.Row():
            response_time = gr.Label(value="Last response time: N/A", label="Response Time")
            cache_status = gr.Label(value="Cache: Not used", label="Cache Status")
    
    # Debug info
    with gr.Accordion("Debug Info", open=False):
        debug_info = gr.Textbox(
            label="Debug Log", 
            value="This panel shows basic information about the conversation.",
            lines=4,
            interactive=False
        )
        
        # Simpler debug function that won't cause issues
        def update_debug(history):
            if not history or len(history) < 2:
                return "No messages in history yet."
            try:
                last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
                return f"Last question: {last_user_msg}"
            except:
                return "Error parsing history."
            
    # Handle message submission with streaming
    def respond(message, chat_history, role):
        """Initial response handler that sets up streaming"""
        if not message.strip():
            return chat_history, "", role
        
        # Set up the streaming response
        updated_history, msg, selected_role = handle_prompt(message, chat_history, role)
        
        # Return the updated history with the thinking placeholder, and clear the message input
        return updated_history, "", selected_role
    
    # Connect the UI components to their respective handlers
    msg.submit(respond, [msg, chatbot, role_dropdown], [chatbot, msg, role_dropdown]).then(
        stream_response, [chatbot, msg, role_dropdown], [chatbot]
    )
    
    submit_btn.click(respond, [msg, chatbot, role_dropdown], [chatbot, msg, role_dropdown]).then(
        stream_response, [chatbot, msg, role_dropdown], [chatbot]
    )
    
    # Update debug info when chatbot changes
    chatbot.change(update_debug, chatbot, debug_info)

app = gr.mount_gradio_app(app, interface, path="/chat")
