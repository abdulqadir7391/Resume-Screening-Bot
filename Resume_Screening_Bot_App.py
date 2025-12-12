"""
Resume Screening Bot using NLP & AI
Technologies: Python, spaCy, SentenceTransformers, Streamlit, PyMuPDF, pandas, docx2txt

Single-file Streamlit app. Features:
 - Upload and parse resumes (PDF/DOCX)
 - Extract skills, education, experience
 - Match resumes to job descriptions using semantic similarity
 - Rank resumes by skill-match percentage and semantic similarity
 - Display match score and skill comparison chart
 - Filter by role (Data Analyst, ML Engineer, etc.)
 - Generate CSV report with candidate rankings and downloadable link
 - Save top resumes to a `top_resumes/` folder
 - (Bonus) Small chatbot-style recommender that suggests resume improvements

Notes before running:
 - Install dependencies: pip install -r requirements.txt
 - Example requirements.txt:
    streamlit
    pandas
    numpy
    spacy
    sentence-transformers
    pymupdf
    python-docx
    docx2txt
    scikit-learn

 - You must download a spaCy model, e.g.:
    python -m spacy download en_core_web_sm

Run:
    streamlit run Resume_Screening_Bot_App.py

"""

import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import os
import re
import io
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import base64
import zipfile

# ----------------------
# Config / Init
# ----------------------
st.set_page_config(layout="wide", page_title="Resume Screening Bot")

MODEL_NAME = "all-MiniLM-L6-v2"  # compact ST model
EMBEDDER = SentenceTransformer(MODEL_NAME)
NLP = spacy.load("en_core_web_sm")

SKILL_KEYWORDS = {
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "pandas", "numpy", "matplotlib", "seaborn", "statistics", "etl"],
    "ML Engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "mlflow", "deep learning", "neural network", "keras", "docker"],
    "Software Engineer": ["java", "python", "c++", "git", "docker", "kubernetes", "rest api"],
}

TOP_FOLDER = Path("top_resumes")
TOP_FOLDER.mkdir(exist_ok=True)

# ----------------------
# Helper functions
# ----------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text("text")
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    # docx2txt expects a path; write to temp
    tmp_path = Path("temp_doc.docx")
    tmp_path.write_bytes(file_bytes)
    text = docx2txt.process(str(tmp_path)) or ""
    try:
        tmp_path.unlink()
    except Exception:
        pass
    return text


def parse_resume(uploaded_file) -> str:
    """Reads uploaded file (BytesIO) and returns text."""
    name = uploaded_file.name.lower()
    bytes_data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(bytes_data)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(bytes_data)
    else:
        # try decode
        try:
            return bytes_data.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def extract_skills_from_text(text: str, role_keywords=None) -> list:
    """Extract skills by keyword matching and simple noun chunk heuristics."""
    text_low = text.lower()
    skills = set()
    # keyword match
    if role_keywords:
        for k in role_keywords:
            if k.lower() in text_low:
                skills.add(k)
            # fuzzy: split k into tokens and search
            for token in k.split():
                if token in text_low:
                    skills.add(k)
    # global keyword search
    for v in SKILL_KEYWORDS.values():
        for sk in v:
            if sk in text_low:
                skills.add(sk)
    # spaCy noun chunks for possible skill phrases (simple heuristic)
    doc = NLP(text)
    for nc in doc.noun_chunks:
        chunk = nc.text.lower().strip()
        if len(chunk) <= 30 and ("experience" not in chunk):
            # keep if contains tech-ish terms
            for kw in ["python","sql","model","tensorflow","pytorch","excel","tableau","power bi"]:
                if kw in chunk:
                    skills.add(chunk)
    return sorted(skills)


def extract_education(text: str) -> list:
    edu = []
    patterns = [r"\b(bachelor[^\n]*)", r"\b(master[^\n]*)", r"\b(b\.?sc[^\n]*)", r"\b(m\.?sc[^\n]*)", r"\b(phd[^\n]*)", r"\b(secondary school|high school|hsc|ssc)" ]
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", m).strip()
            edu.append(cleaned)
    return list(dict.fromkeys([e for e in edu if len(e) > 3]))


def extract_experience_years(text: str) -> float:
    # crude heuristic: look for patterns like 'X years' or date ranges
    yrs = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s+years", text, flags=re.IGNORECASE)
    if m:
        try:
            yrs = float(m.group(1))
            return yrs
        except:
            pass
    # search for date ranges
    years = re.findall(r"(19|20)\d{2}", text)
    if years:
        try:
            years = [int(y) for y in re.findall(r"(19|20)\d{2}", text+" ")]
            # fallback simple heuristic
            if len(years) >= 2:
                return abs(years[-1] - years[0])
        except:
            pass
    # default unknown -> 0
    return yrs


def compute_skill_match(job_skills: list, candidate_skills: list) -> float:
    if not job_skills:
        return 0.0
    js = set([s.lower() for s in job_skills])
    cs = set([s.lower() for s in candidate_skills])
    common = js.intersection(cs)
    return len(common) / len(js) * 100


def embed_text(texts: list) -> np.ndarray:
    # wrapper for embedder
    return EMBEDDER.encode(texts, convert_to_tensor=True)


def semantic_similarity_score(jd_text: str, resume_text: str) -> float:
    emb = EMBEDDER.encode([jd_text, resume_text], convert_to_tensor=True)
    sim = util.cos_sim(emb[0], emb[1]).item()
    return float(sim)


def save_top_resume(file_obj, filename: str):
    path = TOP_FOLDER / filename
    # uploaded file might be SpooledTemporaryFile; reset pointer
    try:
        file_obj.seek(0)
        with open(path, "wb") as f:
            f.write(file_obj.read())
    except Exception:
        # fallback: do nothing
        pass

# ----------------------
# Streamlit UI
# ----------------------

st.title("📝 Resume Screening Bot — NLP & AI")
st.markdown("Upload resumes (PDF/DOCX), paste a job description, and get ranked candidates")

with st.sidebar:
    st.header("Settings")
    role = st.selectbox("Filter by role (optional)", options=["Any"] + list(SKILL_KEYWORDS.keys()))
    top_k = st.number_input("Save top K resumes to folder", min_value=0, max_value=20, value=3)
    save_top = st.checkbox("Save top resumes to disk", value=True)
    download_csv = st.checkbox("Show CSV Download", value=True)

uploaded_files = st.file_uploader("Upload resumes (PDF or DOCX). You can upload multiple.", accept_multiple_files=True)

jd_text = st.text_area("Paste Job Description here", height=220)
if st.button("Auto-fill sample job description"):
    if role != "Any" and role in SKILL_KEYWORDS:
        jd_text = "\n".join(["We are hiring a {} with skills:".format(role)] + SKILL_KEYWORDS[role])
        st.session_state["jd_text"] = jd_text

if "jd_text" in st.session_state and not jd_text:
    jd_text = st.session_state["jd_text"]

# Process uploaded files
results = []
if uploaded_files and jd_text.strip():
    st.info("Processing {} files...".format(len(uploaded_files)))
    job_skills = []
    if role != "Any":
        job_skills = SKILL_KEYWORDS.get(role, [])
    else:
        # simple auto-extract skills from JD using spaCy noun chunks and keyword scan
        job_skills = extract_skills_from_text(jd_text, role_keywords=None)

    jd_emb = embed_text([jd_text])[0]

    for uploaded in uploaded_files:
        text = parse_resume(uploaded)
        candidate_skills = extract_skills_from_text(text, role_keywords=SKILL_KEYWORDS.get(role))
        edu = extract_education(text)
        exp_years = extract_experience_years(text)
        sem_sim = semantic_similarity_score(jd_text, text)
        skill_match_pct = compute_skill_match(job_skills, candidate_skills) if job_skills else 0.0
        combined_score = 0.6 * (skill_match_pct / 100.0) + 0.4 * ((sem_sim + 1) / 2.0)  # normalize sim from [-1,1] to [0,1]
        results.append({
            "filename": uploaded.name,
            "text_snippet": text[:800].replace("\n"," ") + ("..." if len(text)>800 else ""),
            "skills": ", ".join(candidate_skills),
            "education": ", ".join(edu),
            "experience_years": exp_years,
            "skill_match_pct": round(skill_match_pct,2),
            "semantic_similarity": round(sem_sim,4),
            "combined_score": round(combined_score*100,2),
            "file_obj": uploaded,
        })

    # create DataFrame and ranking
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=["combined_score", "skill_match_pct", "semantic_similarity"], ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)

    # Optionally filter by role using content-based heuristic
    if role != "Any":
        # keep resumes that mention at least one role keyword
        role_kw = SKILL_KEYWORDS.get(role, [])
        def role_filter(row):
            txt = row['text_snippet'].lower()
            for kw in role_kw:
                if kw.lower() in txt:
                    return True
            return False
        df_sorted = df_sorted[df_sorted.apply(role_filter, axis=1)]

    # Show results table
    st.subheader("Ranked Candidates")
    st.dataframe(df_sorted[["filename","combined_score","skill_match_pct","semantic_similarity","skills","education","experience_years"]])

    # Display charts for top candidates
    st.subheader("Skill Match Comparison")
    chart_df = df_sorted[["filename","skill_match_pct"]].copy()
    chart_df.rename(columns={"filename":"Candidate","skill_match_pct":"Skill Match (%)"}, inplace=True)
    st.bar_chart(chart_df.set_index("Candidate"))

    # Save top resumes
    if save_top and top_k>0:
        top_rows = df_sorted.head(top_k)
        saved = []
        for idx, row in top_rows.iterrows():
            try:
                save_top_resume(row['file_obj'], row['filename'])
                saved.append(row['filename'])
            except Exception as e:
                st.warning(f"Could not save {row['filename']}: {e}")
        if saved:
            st.success(f"Saved top {len(saved)} resumes to {TOP_FOLDER.resolve()}")

    # CSV report download
    if download_csv:
        csv = df_sorted.drop(columns=["file_obj"]).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f"data:file/csv;base64,{b64}"
        st.markdown(f"[Download CSV Report]({href})")

    # Zip top resumes for download (if saved)
    if save_top and TOP_FOLDER.exists() and any(TOP_FOLDER.iterdir()):
        zip_path = Path("top_resumes.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            for f in TOP_FOLDER.iterdir():
                z.write(f, arcname=f.name)
        with open(zip_path, "rb") as f:
            b64z = base64.b64encode(f.read()).decode()
            st.markdown(f"[Download top resumes ZIP](data:application/zip;base64,{b64z})")

    # Show per-candidate expandable details + recommendations (bonus chatbot-like)
    st.subheader("Candidate Details & Recommendations")
    for idx, row in df_sorted.iterrows():
        with st.expander(f"{row['filename']} — Score: {row['combined_score']}" ):
            st.write("**Snippet:**")
            st.write(row['text_snippet'])
            st.write("**Extracted skills:**", row['skills'])
            st.write("**Education:**", row['education'])
            st.write("**Experience (years):**", row['experience_years'])
            # Simple recommender
            missing = []
            for js in job_skills:
                if js.lower() not in row['skills'].lower():
                    missing.append(js)
            st.write("**Missing job keywords:**", ", ".join(missing) if missing else "None — Good match!")
            recs = []
            if row['skill_match_pct'] < 50:
                recs.append("Add relevant keywords from the job description (skills, tools, and technologies).")
            if row['experience_years'] == 0:
                recs.append("Quantify your experience: add months/years and outcomes (e.g., improved X by Y%).")
            recs.append("Use strong action verbs and metrics for achievements.")
            st.write("**Recommendations:**")
            for r in recs:
                st.write("- ", r)

    # Simple chatbot for targeted suggestions (bonus)
    st.subheader("Resume Advice Chatbot (Bonus)")
    st.write("Ask for improvement tips for a specific candidate. Type the candidate filename or leave 'auto' to pick top candidate.")
    user_query = st.text_input("Candidate filename or 'auto'", value="auto")
    if st.button("Get suggestions"):
        if user_query.strip().lower() == "auto":
            if not df_sorted.empty:
                cand = df_sorted.iloc[0]
            else:
                st.warning("No candidates available")
                cand = None
        else:
            cand = df_sorted[df_sorted['filename']==user_query].squeeze() if not df_sorted.empty else None
            if cand is None or cand.empty:
                st.warning("Candidate not found. Pick 'auto' or check filename.")
                cand = None
        if cand is not None:
            st.write(f"**Suggestions for {cand['filename']}**")
            # generate small textual suggestions
            suggestions = []
            if cand['skill_match_pct'] < 60:
                suggestions.append("Add/Highlight the following missing skills from the job description: " + ", ".join([k for k in job_skills if k.lower() not in cand['skills'].lower()]))
            if cand['education'] == "":
                suggestions.append("Include a concise education section with degree, institution, and year.")
            suggestions.append("Make sure to include measurable outcomes: e.g., 'Reduced processing time by 30% using XYZ'.")
            for s in suggestions:
                st.write("- ", s)

else:
    st.info("Upload resumes and paste a job description to start. You can also choose a role from the sidebar to use role-specific skill keywords.")

# Footer
st.markdown("---")
st.caption("Built with Python, spaCy, SentenceTransformers, PyMuPDF, docx2txt, pandas and Streamlit.")

