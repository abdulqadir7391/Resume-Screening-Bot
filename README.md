# 📝 Resume Screening Bot — NLP & AI

**Resume Screening Bot** is a Streamlit web app that uses NLP and sentence embeddings to automatically parse, score, and rank candidate resumes against a job description. It extracts skills, education, and experience; computes semantic similarity; ranks candidates; and provides recommendations to improve resumes.

---

## 🔧 Technologies
- **Python 3.12 (tested)**  
- **spaCy** (NLP)  
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for semantic embeddings  
- **PyMuPDF** (PDF parsing)  
- **docx2txt / python-docx** (DOCX parsing)  
- **pandas / numpy / scikit-learn** (data handling & metrics)  
- **Streamlit** (UI)

---

## 🚀 Features
- Upload multiple resumes (PDF / DOCX)
- Automatic text extraction
- Extracted fields: **Skills**, **Education**, **Experience (years)**
- **Semantic similarity** between job description and resume
- **Skill match %** and combined ranking score
- Visual **skill comparison chart**
- Filter resumes by **role** (Data Analyst, ML Engineer, Software Engineer)
- Export **CSV** report and download **top resumes** as ZIP
- Bonus: lightweight **chatbot** that suggests resume improvements
- Saves top-K resumes to `top_resumes/`

---

## 📁 Repository structure (suggested)
├─ Resume_Screening_Bot_App.py
├─ requirements.txt
├─ run_app.bat
├─ README.md
├─ .gitignore
├─ top_resumes/ # created at runtime
└─ assets/ # screenshots / demo GIFs

## 🧭 Quick start — Windows (no venv)
> NOTE: installing globally may affect other Python projects. Recommended: use a virtual environment. Steps below show *global* install since requested.

1. Ensure `python` command points to the interpreter you want (Windows example uses Python 3.12):

where python

2. Install dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


3. (Recommended) Prevent transformers from attempting to import TensorFlow:

setx TRANSFORMERS_NO_TF 1
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1

After setx open a new terminal to apply env vars.

4. Run the app:

python -m streamlit run Resume_Screening_Bot_App.py

5. Open the URL shown in the terminal (e.g. http://localhost:8501).


🧩 Troubleshooting tips (Windows)

If ModuleNotFoundError occurs, confirm you installed packages for the same Python executable that runs Streamlit. Use the explicit path to that python.exe if needed:

C:\path\to\python.exe -m pip install -r requirements.txt
C:\path\to\python.exe -m streamlit run Resume_Screening_Bot_App.py


If Hugging Face prints a symlink warning, you can ignore it or run:

setx HF_HUB_DISABLE_SYMLINKS_WARNING 1


If TensorFlow DLL errors appear, disable TF imports for transformers by setting TRANSFORMERS_NO_TF=1 (see above).