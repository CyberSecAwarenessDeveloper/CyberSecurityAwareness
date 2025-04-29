# test_llava.py
import base64
import requests

# Path to a test image
image_path = "temp_image.jpg"  # Use the same image you uploaded

# Convert image to base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Create payload for Ollama
# In test_llava.py
payload = {
    "model": "llava:7b-v1.6-mistral-q4_0",  # Correct model name
    "messages": [
        {
            "role": "user",
            "content": "Is this a phishing email?",
            "images": [base64_image]
        }
    ]
}


# Send request
response = requests.post("http://localhost:11434/api/chat", json=payload)
print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")
