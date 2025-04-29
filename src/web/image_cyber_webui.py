# src/web/image_cyber_webui.py
import gradio as gr, logging, time, os
from fastapi import FastAPI
from src.ml.predictor import answer_with_rag, check_ollama_running
from src.ml.image_analysis import analyze_image, preload_model, analyze_phishing_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Try to preload LLaVA model at startup
try:
    preload_result = preload_model()
    logger.info(f"Preload model result: {preload_result}")
except Exception as e:
    logger.error(f"Error during model preloading: {e}")

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
    
    # Make a copy of history to avoid modifying the input directly
    new_history = list(history)
    
    # Add user message first with correct format
    new_history.append({"role": "user", "content": user_message})
    
    try:
        # Get response from RAG system
        answer = answer_with_rag(user_message, k=3, role=role)
        
        # Log the assistant's response
        logger.info(f"A: {answer[:50]}..." if answer and len(answer) > 50 else f"A: {answer}")
        
        # Add assistant response
        new_history.append({"role": "assistant", "content": answer if answer else "Sorry, I couldn't generate a response."})
        
    except Exception as e:
        logger.error(f"Error in handle_prompt: {e}", exc_info=True)
        new_history.append({"role": "assistant", "content": "I encountered an error while processing your request."})
    
    # Return the updated history for display
    return new_history

def analyze_security_image(image, question, analysis_type="general"):
    """Analyze an image for security implications with specialized analysis types"""
    if image is None:
        return "Please upload an image to analyze."
    
    # Show processing indicator
    processing_msg = "Processing your image... this may take a few minutes for complex images."
    
    # Save image temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Default prompt with the question
    if not question or question.strip() == "":
        question = "Analyze this image for cybersecurity concerns."
    
    try:
        # Set a maximum processing time for the UI feedback
        start_time = time.time()
        
        # Choose analysis type based on selection
        if analysis_type == "phishing":
            # Use specialized phishing analysis
            result = analyze_phishing_email(temp_path, question)
        else:
            # Use general security analysis
            prompt = f"Analyze this image from a cybersecurity perspective. {question}"
            result = analyze_image(temp_path, prompt)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Image analysis completed in {processing_time:.2f} seconds")
        
        # Add processing time to the result
        if not result.startswith("Error") and not result.startswith("Failed"):
            result += f"\n\n(Analysis completed in {processing_time:.2f} seconds)"
        
        return result
    except Exception as e:
        logger.error(f"Error in analyze_security_image: {e}")
        return f"Failed to analyze the image: {str(e)}"

with gr.Blocks(title="Cybersecurity Awareness Assistant") as interface:
    # Status indicator
    with gr.Row():
        status_indicator = gr.Markdown(
            "🟢 Ollama server is running" if check_ollama_running() else "🔴 Ollama server is offline - Using fallback mode"
        )
    
    gr.Markdown("# Cybersecurity Awareness Assistant")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Text Chat"):
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
            
            # Handle message submission with streaming
            def respond_text(message, chat_history, role):
                """Initial response handler that sets up streaming"""
                if not message.strip():
                    return chat_history, "", role
                
                # Set up the streaming response
                updated_history = handle_prompt(message, chat_history, role)
                
                # Return the updated history with the thinking placeholder, and clear the message input
                return updated_history, "", role
            
            # Connect the UI components to their respective handlers
            msg.submit(respond_text, [msg, chatbot, role_dropdown], [chatbot, msg, role_dropdown])
            submit_btn.click(respond_text, [msg, chatbot, role_dropdown], [chatbot, msg, role_dropdown])
            
        with gr.TabItem("Image Analysis"):
            gr.Markdown("# Analyze Security Images")
            
            # Image input and analysis
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                
            with gr.Row():
                image_question = gr.Textbox(
                    label="Question about the image", 
                    placeholder="e.g. Is this a phishing website?",
                    lines=2
                )
                
            with gr.Row():
                analysis_type = gr.Radio(
                    ["general", "phishing"], 
                    label="Analysis Type", 
                    value="phishing",  # Default to phishing analysis
                    info="Choose specialized analysis for different security concerns"
                )
                
            with gr.Row():
                image_analyze_btn = gr.Button("Analyze", variant="primary")
                image_clear_btn = gr.ClearButton([image_input, image_question], value="Clear")
                
            # Output for image analysis with progress info
            gr.Markdown("""
            **Note:** Image analysis may take 1-5 minutes depending on complexity and GPU resources.
            
            For phishing detection:
            - Upload suspicious emails, login pages, or QR codes
            - Choose "phishing" analysis type for specialized detection
            - Add specific questions about sender authenticity, links, or urgent language
            """)
            image_output = gr.Textbox(
                label="Analysis Result", 
                lines=12,
                placeholder="Analysis results will appear here..."
            )
            
            # Connect image analysis components
            image_analyze_btn.click(
                analyze_security_image,
                inputs=[image_input, image_question, analysis_type],
                outputs=[image_output]
            )
    
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
        
        # Simple debug function
        def update_debug(history):
            if not history or len(history) < 2:
                return "No messages in history yet."
            try:
                last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
                return f"Last question: {last_user_msg}"
            except:
                return "Error parsing history."
            
        # Update debug info when chatbot changes
        chatbot.change(update_debug, chatbot, debug_info)

app = gr.mount_gradio_app(app, interface, path="/chat")
