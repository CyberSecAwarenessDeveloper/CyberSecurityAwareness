# src/ml/external_vision_api.py
import os, requests, logging, json

logger = logging.getLogger(__name__)

# Use an external vision API (replace with actual API key and endpoint)
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")

def analyze_image_with_external_api(image_path, prompt="Analyze this image for cybersecurity risks"):
    """Use external vision API to analyze image"""
    try:
        if not os.path.exists(image_path):
            return "Error: Image file not found"
            
        # Read image file
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Prepare request for external API
        headers = {
            "Authorization": f"Bearer {VISION_API_KEY}",
            "Content-Type": "multipart/form-data"
        }
        
        files = {
            "image": ("image.jpg", image_data),
            "prompt": (None, prompt)
        }
        
        # Send request to external API
        response = requests.post(VISION_API_ENDPOINT, headers=headers, files=files, timeout=30)
        response.raise_for_status()
        
        # Parse and return response
        result = response.json()
        return result["analysis"]
    except Exception as e:
        logger.error(f"Error with external vision API: {e}")
        return f"Failed to analyze image: {str(e)}"
