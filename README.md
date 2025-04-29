# Cybersecurity Awareness Assistant: Project Overview & Progress

## Project Summary

This project is an AI-powered Cybersecurity Awareness Assistant designed to help users understand, detect, and prevent cybersecurity threats—especially phishing and social engineering attacks. The system integrates advanced machine learning (ML) models, local large language models (LLMs), and multimodal (text + image) analysis, all running on local infrastructure for privacy and transparency.

---

## Technical Achievements

### ✅ Multi-Model Integration
- **Local ML Pipelines:** Integrated 40+ trained pipelines for threat classification, attack type identification, and incident response.
- **Local LLMs:** DeepSeek-R1:7B for advanced text-based reasoning; LLaVA 7B for image-based (screenshot/email) analysis.
- **RAG (Retrieval-Augmented Generation):** Combines model predictions and document retrieval to enhance LLM answers.

### ✅ Multimodal Security Analysis
- **Image Analysis:** Users can upload screenshots/emails. The system uses OCR (Tesseract) and LLaVA to extract and analyze text, links, QR codes, and visual phishing indicators.
- **Phishing Detection:** Specialized prompts and rule-based checks for urgent language, suspicious domains, QR codes, and more.

### ✅ Robust UI & User Experience
- **Gradio Web Interface:** Interactive chat with role selection (e.g., Security Educator, Analyst).
- **Streaming Responses:** Faster, incremental feedback for user queries.
- **Performance Optimizations:** Caching, model preloading, and adaptive timeouts for better responsiveness.

### ✅ Privacy & Security
- **Local-Only Processing:** No data leaves the user's machine; images and text are analyzed and then deleted.
- **No Cloud Storage:** All analysis is ephemeral unless explicitly saved by the user.

---

## Contextual Achievements

### Societal Impact (LO1)
- **Privacy-First Design:** All analysis is local, respecting user privacy and data legislation (GDPR).
- **Awareness Raising:** Helps users recognize and avoid phishing/social engineering, contributing to a safer digital society.
- **Reflection:** Regularly reviewed ethical implications and updated prompts to avoid bias or hallucination.

### Investigative Problem Solving (LO2)
- **Iterative Development:** Addressed real-world problems (e.g., missed phishing indicators) by adding OCR and rule-based checks.
- **Critical Testing:** Used both synthetic and real phishing emails/screenshots to evaluate and improve detection.

### Data Preparation (LO3)
- **Data Quality:** Collected, cleaned, and transformed multiple cybersecurity datasets (phishing, malware, threat intelligence).
- **Continuous Improvement:** Adjusted data pipelines to improve model accuracy and robustness.

### Machine Teaching (LO4)
- **Model Training:** Trained and evaluated multiple classifiers for different threat categories.
- **Testing & Validation:** Used confusion matrices, accuracy/F1 scores, and real-world test cases.

### Data Visualisation (LO5)
- **Interactive UI:** Users see clear, actionable results and can explore both text and image-based threats.
- **Debug Panel:** Consultants can review system reasoning and performance metrics.

### Reporting (LO6)
- **Documentation:** Maintained clear, methodical documentation of all technical and contextual developments.
- **Transparent Process:** All major design choices and iterations are logged for review.

### Personal Leadership (LO7)
- **Entrepreneurial Mindset:** Proactively sought feedback and implemented advanced features (streaming, multimodal).
- **Self-Reflection:** Regularly assessed learning levels and set concrete improvement goals.

### Personal Goal (LO8)
- *To be described in my PDR after further reflection on my future field of work.*

---

## Project Directory Structure and Contents

This repository contains all code, data, and supporting materials for the Cybersecurity Awareness Assistant project. Below is an overview of the main folders and files:

### Folders

- **.venv/**  
  Python virtual environment for package isolation and reproducibility.

- **data/**  
  Contains knowledge base documents, sample images, and (optionally) cache files for model responses.

- **models/**  
  Stores all trained machine learning pipelines (e.g., `.pkl` files) used for classification and threat detection.

- **src/**  
  Main source code for the project, including:
  - `ml/` – Machine learning logic (model loading, prediction, image analysis, etc.)
  - `web/` – Web UI code (Gradio/FastAPI apps for chat and image analysis)

- **tmp/**  
  Temporary files created during processing (e.g., preprocessed images).

- **Unnecessary_For_Code/**  
  Archive for files not required in the main codebase (for documentation or historical reference).

### Key Files

- **README.md**  
  This documentation file, summarizing the project, progress, and structure.

- **requirements.txt**  
  List of Python dependencies needed to run the project (install with `pip install -r requirements.txt`).

- **setup.py**  
  (Optional) Setup script for packaging or easy installation.

- **test_llava.py**  
  Standalone script for testing the LLaVA model’s image analysis capabilities.

- **test_tesseract.py**  
  Standalone script for verifying Tesseract OCR integration and functionality.

- **temp_image.jpg / temp_image2.png**  
  Example images used for testing image analysis and OCR features.

- **toLookAt**  
  A file for notes, reminders, or items to review further.

- **.gitignore**  
  Specifies files and folders to be ignored by Git version control (e.g., `.venv`, `__pycache__`, etc.).

### How the Contents Work Together

- **Web Interface**: The `src/web` folder contains the Gradio/FastAPI apps that provide both chat and image analysis interfaces for end-users.
- **Machine Learning**: The `src/ml` folder contains all logic for loading models, making predictions, and analyzing images (including OCR and phishing detection).
- **Data and Models**: The `data` and `models` folders provide the necessary inputs for the assistant to function, including both structured datasets and trained pipelines.
- **Testing and Validation**: Scripts like `test_llava.py` and `test_tesseract.py` allow for independent verification of LLaVA and OCR functionality.
- **Documentation and Setup**: The `README.md` and `requirements.txt` ensure that new users and reviewers can understand, install, and run the project efficiently.

---

*This structure ensures that the project is modular, maintainable, and easy to extend or audit for both technical and educational purposes.*

---

## Learning Level Self-Assessment

| LO   | Level      | Reflection |
|------|------------|------------|
| LO1  | Beginning/Proficient | Privacy and societal impact are considered in all design choices. A societal impact analysis will be done|
| LO2  | Proficient | Problems are identified and solved iteratively, with evidence in code and documentation. |
| LO3  | Proficient | Data is cleaned, validated, and pipelines are robust. |
| LO4  | Proficient | Models are trained, tested, and improved based on real feedback. |
| LO5  | Beginning  | UI is clear and functional, but more advanced visualizations could be added. |
| LO6  | Beginning  | Documentation and reporting needs to be up-to-date and thorough. |
| LO7  | Proficient | Demonstrated initiative and continuous improvement mindset. |
| LO8  | TBA        | Will be set in the PDR. |

---

## Development Level Scale

The project is currently at the **Proficient** level for most LOs, with clear evidence of progress and reflection. The system is robust, privacy-focused, and integrates state-of-the-art techniques for cybersecurity awareness. Continuous improvements are being made, especially in visualization and user experience.

---

## Questions for Reviewers

- Are there additional indicators or features you would recommend for phishing detection, especially in multilingual contexts?
- Would you like to see more advanced data visualizations or reporting features?
- Are there ethical or privacy concerns I should address further as the project evolves?
- Is there a particular LO where you feel I should focus more effort before the final assessment?

---

*This README is intended to provide a clear, comprehensive overview for teachers and consultants. Please refer to the codebase and documentation for further technical details.*
