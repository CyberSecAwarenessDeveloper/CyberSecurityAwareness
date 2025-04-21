# src/ml/load_models.py
import os, joblib, logging
logger = logging.getLogger(__name__)

def load_all_models(model_dir="models/trained_pipelines"):
    """Load all trained model pipelines from the specified directory"""
    models = {}
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        for fname in os.listdir(model_dir):
            if fname.endswith(".pkl"):
                path = os.path.join(model_dir, fname)
                try:
                    models[fname.replace(".pkl", "")] = joblib.load(path)
                    logger.info(f"✅ loaded {fname}")
                except Exception as e:
                    logger.error(f"❌ {fname}: {e}")
        
        if not models:
            logger.warning(f"No model files (.pkl) found in {model_dir}")
            
    except FileNotFoundError:
        logger.warning(f"Directory {model_dir} not found. No models loaded.")
        
    return models
