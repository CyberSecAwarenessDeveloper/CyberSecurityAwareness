# src/ml/predictor.py
import os, logging, requests, json, re, time
import numpy as np
from collections import namedtuple
from src.ml.load_models import load_all_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODELS = load_all_models()  # {filename: sklearn‑pipeline}
Label = namedtuple("Label", ["conf", "model_key", "label"])

# Role-specific system prompts
ROLE_PROMPTS = {
    "Security Expert": "You are a cybersecurity expert with deep technical knowledge. Provide detailed technical explanations about security concepts and threats.",
    "Security Educator": "You are a cybersecurity educator who explains complex concepts simply. Focus on clear explanations for non-technical users. DO NOT include <think></think> tags in your response.",
    "Security Analyst": "You are a security analyst who investigates threats. Focus on analytical approaches and detection techniques.",
    "Security Advisor": "You are a security advisor who provides practical recommendations. Focus on actionable advice and best practices."
}

# Default role
DEFAULT_ROLE = "Security Expert"

# ------------------------------------------------------------------ #
# 1.   TOP‑k LABELS FROM ALL MODELS
# ------------------------------------------------------------------ #
def get_top_labels(text: str, k: int = 3) -> list[Label]:
    scored: list[Label] = []
    for name, model in MODELS.items():
        try:
            proba = model.predict_proba([text])[0]
            idx = proba.argmax()
            scored.append(Label(float(proba[idx]), name, model.classes_[idx]))
        except Exception as e:
            logger.debug(f"Model {name} prediction failed: {e}")
            continue
    scored.sort(reverse=True, key=lambda x: x.conf)
    return scored[:k]

# ------------------------------------------------------------------ #
# 2.   DOCUMENT RETRIEVAL
# ------------------------------------------------------------------ #
def retrieve_documents(query: str, k: int = 3) -> list[dict]:
    """Simple keyword-based retrieval from text files"""
    kb_dir = "data/kb"
    os.makedirs(kb_dir, exist_ok=True)
    
    # Create sample docs if directory is empty
    if not os.listdir(kb_dir):
        create_sample_documents(kb_dir)
    
    docs = []
    query_terms = set(query.lower().split())
    
    if not query_terms:
        return []
    
    for fname in os.listdir(kb_dir):
        if fname.endswith((".txt", ".md")):
            try:
                with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
                    content = f.read()
                    
                # Simple relevance scoring
                doc_terms = set(content.lower().split())
                matching_terms = query_terms.intersection(doc_terms)
                if matching_terms:
                    relevance_score = len(matching_terms) / len(query_terms)
                    docs.append({
                        "filename": fname,
                        "content": content[:800],  # Truncate long docs
                        "score": relevance_score
                    })
            except Exception as e:
                logger.error(f"Error reading {fname}: {e}")
    
    # Sort by relevance and return top k
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:k]

def create_sample_documents(kb_dir):
    """Create basic cybersecurity knowledge documents"""
    topics = {
        "ddos.txt": """
A Distributed Denial of Service (DDoS) attack is a malicious attempt to disrupt the normal traffic of a targeted server, service, or network by overwhelming the target or its surrounding infrastructure with a flood of Internet traffic. DDoS attacks utilize multiple compromised computer systems as sources of attack traffic. Exploited machines can include computers and other networked resources such as IoT devices. 

Common types of DDoS attacks include volume-based attacks (UDP floods), protocol attacks (SYN floods), and application layer attacks (HTTP floods). Mitigation strategies include traffic filtering, CDNs, load balancing, and DDoS protection services.
        """,
        
        "phishing.txt": """
Phishing is a type of social engineering attack often used to steal user data, including login credentials and credit card numbers. It occurs when an attacker, masquerading as a trusted entity, dupes a victim into opening an email, instant message, or text message. The recipient is then tricked into clicking a malicious link, which can lead to the installation of malware or the revealing of sensitive information.

Common types include email phishing, spear phishing (targeted attacks), and smishing (SMS phishing). Prevention measures include verifying sender addresses, avoiding suspicious links, and using multi-factor authentication.
        """,
        
        "ransomware.txt": """
Ransomware is a type of malicious software designed to block access to a computer system until a sum of money is paid. Ransomware typically spreads through phishing emails or by a victim unknowingly visiting an infected website. Once executed in the system, ransomware encrypts files, rendering them inaccessible, or locks the system.

Prevention measures include regular backups, security awareness training, keeping systems updated, using anti-malware protection, and network segmentation.
        """,
        
        "malware.txt": """
Malware is malicious software designed to damage, disrupt, or gain unauthorized access to computer systems. Common types include viruses, worms, trojans, spyware, adware, and ransomware.

Viruses attach to legitimate programs and spread when those programs run. Worms self-replicate without user action. Trojans disguise themselves as legitimate software to trick users into installing them. Spyware collects information without consent. Adware displays unwanted advertisements. Ransomware encrypts files and demands payment for decryption.

Prevention includes using antivirus software, keeping systems updated, being cautious with email attachments and downloads, and implementing regular backups.
        """
    }
    
    for filename, content in topics.items():
        with open(os.path.join(kb_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
    
    logger.info(f"✅ Created {len(topics)} sample documents in knowledge base")

# ------------------------------------------------------------------ #
# 3.   RESPONSE SANITIZATION
# ------------------------------------------------------------------ #
def sanitize_response(text):
    """Clean up response to avoid Gradio display issues"""
    if not isinstance(text, str):
        return "Error: Unable to generate a proper response"
    
    if not text.strip():
        return "I don't have enough information to answer that question properly."
    
    # Remove <think>...</think> tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Fix any problematic characters or patterns
    text = text.replace('\u0000', '')  # Remove null bytes
    
    return text.strip()

# ------------------------------------------------------------------ #
# 4.   FALLBACK RESPONSE FOR OFFLINE MODE
# ------------------------------------------------------------------ #
def get_fallback_response(question: str) -> str:
    """Provide a fallback response when Ollama is unavailable"""
    # Generic cybersecurity fallback
    if "ddos" in question.lower() or "denial" in question.lower():
        return """
A DDoS (Distributed Denial of Service) attack is a type of cyberattack where attackers flood a target server or network with excessive traffic from multiple sources, making it unavailable to users.

Key characteristics:
- Uses multiple compromised computers (often forming a botnet)
- Overwhelms network resources and bandwidth
- Causes service disruption for legitimate users
- Can target websites, online services, or entire networks

Common defenses include traffic filtering, rate limiting, using CDNs, and DDoS protection services.
        """
    elif "phishing" in question.lower():
        return """
Phishing is a cybersecurity attack where criminals impersonate trusted entities to trick people into revealing sensitive information like passwords or credit card details.

Phishing typically uses deceptive emails, fake websites, or messages that appear legitimate but are designed to steal your information.
        """
    else:
        return """
I'm currently operating in offline mode because the AI model isn't available. I can provide basic information about common cybersecurity topics like DDoS attacks, phishing, malware, and best practices.

Please try asking a more specific question about a cybersecurity topic, or try again later when the connection to the AI model is restored.
        """

# ------------------------------------------------------------------ #
# 5.   OLLAMA CONNECTION WITH IMPROVED ERROR HANDLING
# ------------------------------------------------------------------ #
OLLAMA_URL = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:7b")

def preload_model():
    """Preload the model to avoid cold start delays"""
    try:
        logger.info(f"Preloading Ollama model: {OLLAMA_MODEL}")
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Hello, this is a preload test."},
                {"role": "user", "content": "Hi"}
            ],
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_gpu": 1,
                "num_thread": 8,
                "temperature": 0.7,
                "repeat_penalty": 1.1,
                "keep_alive": "1h"  # Keep model loaded for 1 hour
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(120, 120))
        r.raise_for_status()
        logger.info(f"✅ Model {OLLAMA_MODEL} preloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {e}")
        return False

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        return r.status_code == 200
    except:
        return False

def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Send request to Ollama with better error handling"""
    # First check if Ollama is actually running
    if not check_ollama_running():
        logger.error("Ollama server is not running")
        return get_fallback_response(user_prompt)
    
    for attempt in range(2):  # Try twice
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "num_gpu": 1,
                    "num_thread": 8,
                    "temperature": 0.7,
                    "repeat_penalty": 1.1,
                    "keep_alive": "1h"
                }
            }
            
            # Longer timeouts (120s connection, 300s read)
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(120, 300))
            r.raise_for_status()
            
            response_json = r.json()
            if "message" not in response_json or "content" not in response_json["message"]:
                raise ValueError(f"Unexpected response format from Ollama")
                
            content = response_json["message"]["content"].strip()
            
            # Make sure we don't expose thinking process
            cleaned_content = sanitize_response(content)
            
            if not cleaned_content:
                raise ValueError("Received empty response from model")
                
            return cleaned_content
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout (attempt {attempt+1}/2)")
            if attempt == 1:
                return get_fallback_response(user_prompt)
        except Exception as e:
            logger.error(f"Ollama error (attempt {attempt+1}/2): {e}")
            if attempt == 1:
                return get_fallback_response(user_prompt)
    
    return get_fallback_response(user_prompt)

# ------------------------------------------------------------------ #
# 6.   PUBLIC ENTRYPOINT FOR THE UI
# ------------------------------------------------------------------ #
def answer_with_rag(question: str, k: int = 3, role: str = DEFAULT_ROLE) -> str:
    """Generate answer using RAG with multiple models and document retrieval"""
    try:
        # Get model predictions
        labels = get_top_labels(question, k)
        
        # Retrieve relevant documents
        documents = retrieve_documents(question, k=k)
        
        # Construct document context
        doc_context = ""
        if documents:
            for i, doc in enumerate(documents):
                doc_context += f"\n\nDocument {i+1} ({doc['filename']}):\n{doc['content']}"
        else:
            doc_context = "\n\nNo relevant documents found in the knowledge base."
        
        # Format model insights
        model_insights = ""
        if labels:
            insights = []
            for label in labels:
                insights.append(f"- {label.label} (from {label.model_key}, confidence: {label.conf:.2f})")
            model_insights = "\n".join(insights)
        else:
            model_insights = "No confident model predictions available."
        
        # Construct RAG prompt with explicit instructions not to show thinking
        rag_prompt = (
            f"User question: {question}\n\n"
            f"Relevant cybersecurity information:{doc_context}\n\n"
            f"Model classification insights:\n{model_insights}\n\n"
            "Based on the above information, provide a comprehensive answer "
            "to the user's question about cybersecurity. Cite specific information when available. "
            "DO NOT include <think></think> tags or any preliminary thinking process in your response."
        )
        
        # Get system prompt based on role
        system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS[DEFAULT_ROLE])
        system_prompt += " DO NOT include any thinking or planning process in your response."
        
        # Get answer from LLM
        answer = ask_llm(system_prompt, rag_prompt)
        
        # Return sanitized answer
        return sanitize_response(answer)
    except Exception as e:
        logger.error(f"Error in answer_with_rag: {e}", exc_info=True)
        return "I'm sorry, but I encountered an error processing your question. Please try again later."
