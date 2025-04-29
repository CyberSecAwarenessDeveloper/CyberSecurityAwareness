import pytesseract
from PIL import Image

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path

# Print Tesseract version to verify it's working
print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

# Test with a simple image if you have one
try:
    text = pytesseract.image_to_string(Image.open("temp_image2.png"))
    print("OCR result:", text[:100], "...")  # Print first 100 chars
except Exception as e:
    print(f"Error: {e}")
