# Cybersecurity Awareness Assistant: Project Overview & Progress
## Project Summary
This project is an AI-powered Cybersecurity Awareness Assistant
designed to help users understand, detect, and prevent
cybersecurity threats Multi-Model Integration
- **Local ML Pipelines:** Integrated 40+ trained pipelines for
threat classification, attack type identification, and incident
response.
- **Local LLMs:** DeepSeek-R1:7B for advanced text-based
reasoning; LLaVA 7B for image-based (screenshot/email) analysis.
- **RAG (Retrieval-Augmented Generation):** Combines model
predictions and document retrieval to enhance LLM answers.
### Robust UI & User Experience
- **Gradio Web Interface:** Interactive chat with role selection
(e.g., Security Educator, Analyst).
- **Streaming Responses:** Faster, incremental feedback for user
queries.
- **Performance Optimizations:** Caching, model preloading, and
adaptive timeouts for better responsiveness.
### Machine learning logic (model loading, prediction, image
analysis, etc.)
- `web/` s image analysis capabilities.
- **test_tesseract.py**
Standalone script for verifying Tesseract OCR integration and
functionality.
- **temp_image.jpg / temp_image2.png**
Example images used for testing image analysis and OCR features.
- **toLookAt**
A file for notes, reminders, or items to review further.
- **.gitignore**
Specifies files and folders to be ignored by Git version control
(e.g., `.venv`, `__pycache__`, etc.).
### How the Contents Work Together
- **Web Interface**: The `src/web` folder contains the
Gradio/FastAPI apps that provide both chat and image analysis
interfaces for end-users.
- **Machine Learning**: The `src/ml` folder contains all logic
for loading models, making predictions, and analyzing images
(including OCR and phishing detection).
- **Data and Models**: The `data` and `models` folders provide
the necessary inputs for the assistant to function, including
both structured datasets and trained pipelines.
- **Testing and Validation**: Scripts like `test_llava.py` and
`test_tesseract.py` allow for independent verification of LLaVA
and OCR functionality.
- **Documentation and Setup**: The `README.md` and
`requirements.txt` ensure that new users and reviewers can
understand, install, and run the project efficiently.
---
*This structure ensures that the project is modular,
maintainable, and easy to extend or audit for both technical and
educational purposes.*
---
## Learning Level Self-Assessment
| LO | Level | Reflection |
|------|------------|------------|
| LO1 | Beginning/Proficient | Privacy and societal impact are considered in all design choices. A societal impact analysis will be done|
| LO2 | Proficient | Problems are identified and solved iteratively, with evidence in code and documentation. |
| LO3 | Proficient | Data is cleaned, validated, and pipelines are robust. |
| LO4 | Proficient | Models are trained, tested, and improved based on real feedback. |
| LO5 | Beginning | UI is clear and functional, but more advanced visualizations could be added. |
| LO6 | Beginning | Documentation and reporting needs to be up- to-date and thorough. |
| LO7 | Proficient | Demonstrated initiative and continuous improvement mindset. |
| LO8 | TBA | Will be set in the PDR. |
---
## Development Level Scale
The project is currently at the **Proficient** level for most
LOs, with clear evidence of progress and reflection. The system
is robust, privacy-focused, and integrates state-of-the-art
techniques for cybersecurity awareness. Continuous improvements
are being made, especially in visualization and user experience.
---
## Questions for Reviewers
- Are there additional indicators or features you would recommend
for phishing detection, especially in multilingual contexts?
- Would you like to see more advanced data visualizations or
reporting features?
- Are there ethical or privacy concerns I should address further
as the project evolves?
- Is there a particular LO where you feel I should focus more
effort before the final assessment?
---
*This README is intended to provide a clear, comprehensive
overview for teachers and consultants. Please refer to the
codebase and documentation for further technical details.*
