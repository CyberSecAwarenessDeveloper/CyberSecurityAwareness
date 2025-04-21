# src/web/cyber_webui.py
import gradio as gr, logging
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
    
    # Create new history list (don't modify input directly)
    new_history = list(history)
    
    # Add user message to history
    new_history.append({"role": "user", "content": user_message})
    
    try:
        # Get response from RAG system
        answer = answer_with_rag(user_message, k=3, role=role)
        
        # Log the response (truncated)
        logger.info(f"A: {answer[:50]}..." if answer and len(answer) > 50 else f"A: {answer}")
        
        # Add the assistant's response to history
        new_history.append({"role": "assistant", "content": answer})
        
    except Exception as e:
        logger.error(f"Error in handle_prompt: {e}", exc_info=True)
        new_history.append({"role": "assistant", "content": "I encountered an error while processing your request."})
    
    # Return the new history
    return new_history

with gr.Blocks(title="Cybersecurity Awareness Assistant") as interface:
    # Status indicator
    with gr.Row():
        ollama_status = gr.State(check_ollama_running())
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
    
    # Chat interface with messages format - NO AVATARS
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
    
    def respond(message, chat_history, role):
        """Process a user message and return the updated chat history"""
        if not message.strip():
            return chat_history, ""
            
        # Process the message and get updated history
        updated_history = handle_prompt(message, chat_history, role)
        
        # Return updated history and clear message box
        return updated_history, ""

    # Connect UI components to functions
    msg.submit(respond, [msg, chatbot, role_dropdown], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot, role_dropdown], [chatbot, msg])
    
    # Debug panel with basic info only
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
            
        chatbot.change(update_debug, chatbot, debug_info)

app = gr.mount_gradio_app(app, interface, path="/chat")
