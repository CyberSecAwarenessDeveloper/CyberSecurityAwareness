# src/ml/lightweight_vision.py
import os, torch, logging
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

# Use a small, specialized model focused only on cybersecurity-relevant features
VISION_MODEL = "microsoft/resnet-50"  # Replace with an actual cybersecurity-focused model if available
LABELS = ["phishing_website", "legitimate_website", "malware_screenshot", "security_certificate", "data_leak"]

class LightweightVisionAnalyzer:
    def __init__(self):
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(VISION_MODEL)
            self.model = AutoModelForImageClassification.from_pretrained(VISION_MODEL)
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            self.initialized = False
    
    def analyze(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Return predicted class and confidence scores
        # ...

            
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get predicted class
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.model.config.id2label[predicted_class_idx]
            
            # Get confidence scores for each class
            scores = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
            results = {
                self.model.config.id2label[i]: score
                for i, score in enumerate(scores)
            }
            
            # Format response
            response = f"Image classified as: {predicted_class}\n\nConfidence scores:\n"
            for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
                response += f"- {label}: {score:.2f}\n"
                
            return response
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return f"Failed to analyze image: {str(e)}"
