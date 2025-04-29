# src/ml/image_analysis.py
import os, logging, requests, base64, time, re
from PIL import Image
from datetime import datetime
import io
import pytesseract
# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path if needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
OLLAMA_URL = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
MULTIMODAL_MODEL = "llava:7b-v1.6-mistral-q4_0"

# Specialized system prompts for different analysis types
PHISHING_SYSTEM_PROMPT = """
You are a cybersecurity expert specializing in phishing detection with over 15 years of experience.
Your task is to analyze the provided image with extreme scrutiny for signs of phishing.

Pay special attention to:
1. Sender email domain inconsistencies (e.g., institutional emails from gmail accounts)
2. Urgent calls to action or threatening language
3. Suspicious links, buttons, or QR codes for authentication
4. Brand/institutional impersonation
5. Grammar, spelling, or formatting errors
6. Suspicious date markers or future dates
7. Requests for sensitive information or immediate action

Be EXTREMELY suspicious of:
- Emails asking users to scan QR codes for authentication
- Emails claiming to be from institutions but sent from personal domains
- Security alerts requiring immediate action
- Emails with misaligned or low-quality logos
"""

def preprocess_image(image_path, max_size=(800, 800)):
    """Enhanced preprocessing with better quality preservation"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (removes alpha channel if present)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Better resize method with higher quality preservation
            if img.width > max_size[0] or img.height > max_size[1]:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(max_size[0] / img.width, max_size[1] / img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized_img = img
            
            # Save with higher quality
            preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
            resized_img.save(preprocessed_path, format="JPEG", quality=95)
            logger.info(f"Preprocessed image saved to {preprocessed_path}")
            return preprocessed_path
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image_path  # Return original path if processing fails

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR"""
    try:
        # Import pytesseract with explicit path
        import pytesseract
        # Set the tesseract executable path (adjust the path according to your installation)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Open and process the image
        from PIL import Image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Extract text
            text = pytesseract.image_to_string(img)
            logger.info(f"Successfully extracted {len(text)} characters of text from image")
            return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def detect_qr_code(image_path):
    """Detect if an image contains a QR code"""
    try:
        # Try to import pyzbar - requires installation
        try:
            from pyzbar.pyzbar import decode
            
            # Open and decode image
            with Image.open(image_path) as img:
                decoded_objects = decode(img)
                # Return True if any QR code is found
                for obj in decoded_objects:
                    if obj.type == 'QRCODE':
                        logger.info("QR code detected in image")
                        return True
                return False
        except ImportError:
            logger.warning("pyzbar not installed. QR detection disabled.")
            # Assume QR might be present if "QR" is in the filename as fallback
            return "qr" in image_path.lower()
    except Exception as e:
        logger.error(f"QR detection failed: {e}")
        return False

def check_phishing_indicators(image_path, extracted_text):
    """Check for common phishing indicators in image and text"""
    indicators = []
    
    # 1. Check for suspicious email domains impersonating institutions
    if (re.search(r'university|college|edu|bank|account|security', extracted_text, re.I) and 
        re.search(r'@gmail\.com|@yahoo\.com|@hotmail\.com|@outlook\.com', extracted_text, re.I)):
        indicators.append("Institutional communication sent from personal email domain")
    
    # 2. Check for urgent language
    if re.search(r'urgent|immediate|alert|warning|limited time|expires|terminate|suspended', extracted_text, re.I):
        indicators.append("Contains urgent/threatening language")
    
    # 3. Check for QR code in emails (high-risk indicator)
    if re.search(r'scan|qr code|verify|authenticate', extracted_text, re.I):
        # Verify if image actually contains a QR code
        if detect_qr_code(image_path):
            indicators.append("QR code for authentication (high-risk phishing indicator)")
    
    # 4. Check for suspicious future dates
    current_year = datetime.now().year
    future_year_pattern = '|'.join([str(y) for y in range(current_year + 1, current_year + 3)])
    if re.search(future_year_pattern, extracted_text):
        indicators.append("Email contains future dates")
        
    # 5. Check for suspicious links
    if re.search(r'click here|login|sign in|verify|confirm', extracted_text, re.I):
        indicators.append("Contains calls to click links or verify/confirm credentials")
    
    return indicators

def encode_image_to_base64(image_path):
    """Convert image to base64 for Ollama API"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def preload_model():
    """Preload the model with keep_alive to prevent unloading"""
    try:
        logger.info(f"Preloading model: {MULTIMODAL_MODEL} with keep_alive setting")
        # Simple prompt to load the model
        payload = {
            "model": MULTIMODAL_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, this is a test to preload the model."}
            ],
            "options": {
                "keep_alive": "1h"  # Keep model loaded for an hour
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(10, 60))
        r.raise_for_status()
        logger.info(f"✅ Model {MULTIMODAL_MODEL} preloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {e}")
        return False

def analyze_image(image_path, prompt="", is_phishing_analysis=False, include_ocr=True):
    """Analyze image using a multimodal model through Ollama with enhanced error handling"""
    start_time = time.time()
    
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Preprocess the image to optimize for analysis
        preprocessed_path = preprocess_image(image_path)
        logger.info(f"Analyzing image at path: {preprocessed_path}")
        
        # Extract text if OCR is enabled
        extracted_text = ""
        if include_ocr:
            extracted_text = extract_text_from_image(preprocessed_path)
        
        # Check for phishing indicators if this is a phishing analysis
        phishing_indicators = []
        if is_phishing_analysis:
            phishing_indicators = check_phishing_indicators(preprocessed_path, extracted_text)
            
        # Enhance the prompt with extracted text and indicators
        enhanced_prompt = prompt
        if extracted_text:
            enhanced_prompt += f"\n\nExtracted text from image:\n{extracted_text}"
        if phishing_indicators:
            enhanced_prompt += f"\n\nSuspicious elements detected:\n" + "\n".join(f"- {ind}" for ind in phishing_indicators)
        
        # Choose appropriate system prompt based on analysis type
        system_content = PHISHING_SYSTEM_PROMPT if is_phishing_analysis else "You are a cybersecurity expert. Analyze this image carefully."
            
        # Encode image to base64
        base64_image = encode_image_to_base64(preprocessed_path)
        if not base64_image:
            return "Error: Failed to encode image"
        
        # Prepare payload for Ollama with explicit keep_alive
        payload = {
            "model": MULTIMODAL_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": enhanced_prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_gpu": 1,
                "num_thread": 8,
                "temperature": 0.2,  # Lower temperature for more focused responses
                "keep_alive": "1h"  # Keep model loaded for an hour
            }
        }
        
        logger.info(f"Sending request to Ollama with model: {MULTIMODAL_MODEL}")
        
        # First try a ping to verify Ollama is responsive
        try:
            ping = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
            ping.raise_for_status()
            logger.info(f"Ollama server is responding: {ping.json()}")
        except Exception as e:
            logger.error(f"Ollama server not responding: {e}")
            return "Error: Ollama server not responding. Please ensure Ollama is running."
        
        # Send request with progressive timeout handling
        timeouts = [(10, 60), (20, 120), (30, 300)]  # Multiple attempts with increasing timeouts
        
        for attempt, (connect_timeout, read_timeout) in enumerate(timeouts, 1):
            try:
                logger.info(f"Attempt {attempt} with timeout ({connect_timeout}, {read_timeout})")
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat", 
                    json=payload, 
                    timeout=(connect_timeout, read_timeout)
                )
                r.raise_for_status()
                
                # Process response
                response_json = r.json()
                if "message" not in response_json or "content" not in response_json["message"]:
                    raise ValueError(f"Unexpected response format from Ollama: {response_json}")
                    
                content = response_json["message"]["content"].strip()
                
                # Log success and timing
                elapsed = time.time() - start_time
                logger.info(f"Successfully received response in {elapsed:.2f} seconds")
                
                # Clean up temporary file if we created one
                if preprocessed_path != image_path and os.path.exists(preprocessed_path):
                    try:
                        os.remove(preprocessed_path)
                    except:
                        pass
                        
                # Add phishing indicators if any were found
                if phishing_indicators:
                    content += "\n\n**Suspicious elements detected:**\n" + "\n".join(f"- {ind}" for ind in phishing_indicators)
                    
                return content
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt}/{len(timeouts)}")
                if attempt == len(timeouts):
                    # Final attempt failed
                    # Return a useful response with any detected indicators
                    if phishing_indicators:
                        return ("Analysis timed out, but suspicious elements were detected:\n" + 
                                "\n".join(f"- {ind}" for ind in phishing_indicators) +
                                "\n\nThe image may be too complex or the server is under heavy load.")
                    else:
                        return "Analysis timed out. The image may be too complex or the server is under heavy load."
            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {e}")
                if attempt == len(timeouts):
                    if phishing_indicators:
                        return ("Failed to complete full analysis, but suspicious elements were detected:\n" + 
                                "\n".join(f"- {ind}" for ind in phishing_indicators) +
                                f"\n\nError: {str(e)}")
                    else:
                        return f"Failed to analyze image: {str(e)}"
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return f"Failed to analyze image: {str(e)}"

def analyze_phishing_email(image_path, question=""):
    """Specialized function for phishing email analysis"""
    # Craft a targeted prompt for phishing detection
    prompt = (
        "This image may contain a phishing email or message. " +
        "Analyze it with extreme scrutiny to determine if it's a phishing attempt. "
    )
    
    if question:
        prompt += f"\n\nUser's question: {question}"
        
    prompt += """
    
    For your analysis, please consider and explicitly address:
    1. Sender legitimacy: Check domain names and email addresses for inconsistencies
    2. Urgency tactics: Note any pressure to act quickly
    3. Authentication methods: Be highly suspicious of QR codes for authentication
    4. Links or buttons: Are they directing to legitimate sites?
    5. Branding accuracy: Are logos and formatting consistent with official communications?
    6. Language quality: Note any grammatical errors or unusual phrasing
    
    If this is a phishing attempt, also explain what the attackers are trying to accomplish.
    """
    
    # Use the enhanced image analysis with phishing-specific settings
    return analyze_image(image_path, prompt, is_phishing_analysis=True, include_ocr=True)
