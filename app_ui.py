# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import os
import zipfile
import re
import numpy as np
import time

from pdfminer.high_level import extract_text
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


# ============================================================
# 🎨 CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d2ffff 0%, #a1a8f7 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}
.stButton>button {
    background-color: #6c63ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #574fd6;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("📊 Resume Screening & Skill Matching Dashboard")
st.markdown("### Upload resumes + enter job description → get ranked candidates")
st.markdown("---")


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("⚙️ Input Panel")

job_description = st.sidebar.text_area(
    "📝 Job Description",
    height=200,
    placeholder="Paste job description here..."
)

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Resume ZIP",
    type=["zip"]
)

run_button = st.sidebar.button("🚀 Run Matching")


# ============================================================
# FUNCTIONS
# ============================================================

def extract_resume_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        try:
            text = extract_text(file_path)
        except:
            pass
    elif file_path.endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            text = " ".join([p.text for p in doc.paragraphs])
        except:
            pass
    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", errors="ignore") as f:
                text = f.read()
        except:
            pass
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    return text


# ============================================================
# MAIN
# ============================================================

if run_button:

    if uploaded_file is None or job_description.strip() == "":
        st.error("⚠️ Please upload resumes and enter job description")
        st.stop()

    start_time = time.time()

    with st.spinner("🔄 Processing resumes..."):

        # CLEAN OLD DATA
        if os.path.exists("data"):
            import shutil
            shutil.rmtree("data")

        # SAVE ZIP
        with open("data.zip", "wb") as f:
            f.write(uploaded_file.read())

        # EXTRACT
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall("data")

        # FIND ROOT FOLDER
        DATA_DIR = "data"
        while True:
            items = os.listdir(DATA_DIR)
            if all(os.path.isdir(os.path.join(DATA_DIR, item)) for item in items):
                break
            elif len(items) == 1:
                DATA_DIR = os.path.join(DATA_DIR, items[0])
            else:
                break

        # LOAD FILES
        resume_texts = []
        resume_ids = []
        all_files = []

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                all_files.append(os.path.join(root, file))

        progress = st.progress(0)

        for i, path in enumerate(all_files):
            text = extract_resume_text(path)
            if len(text.strip()) > 0:
                resume_texts.append(text)
                resume_ids.append(path)
            progress.progress((i + 1) / len(all_files))

        if len(resume_texts) == 0:
            st.error("❌ No resumes found")
            st.stop()

        # LOAD TRAIN DATA
        train_df = pd.read_csv("Resume.csv", engine="python", on_bad_lines="skip")
        train_df = train_df[['ID','Resume_str','Category']].dropna()

        train_texts = [clean_text(t) for t in train_df["Resume_str"]]
        train_labels = train_df["Category"].str.upper().values

        # PREPROCESS
        resumes_clean = [clean_text(r) for r in resume_texts]
        jd_clean = clean_text(job_description)

        # TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        vectorizer.fit(train_texts)

        train_vectors = vectorizer.transform(train_texts)
        resume_vectors = vectorizer.transform(resumes_clean)
        jd_vector = vectorizer.transform([jd_clean])[0]

        # MODEL
        target_category = "BANKING"

        y_train = np.array([
            1 if label == target_category else 0
            for label in train_labels
        ])

        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(train_vectors, y_train)

        # SIMILARITY
        similarity_scores = cosine_similarity(
            resume_vectors,
            jd_vector.reshape(1, -1)
        ).flatten()

        sorted_indices = similarity_scores.argsort()[::-1]
        top_k = sorted_indices[:50]

        # SUITABILITY
        probabilities = model.predict_proba(resume_vectors[top_k])[:, 1]

        # BUILD DF
        data = []

        for i, idx in enumerate(top_k):
            similarity = round(similarity_scores[idx] * 100, 2)
            suitability = round(probabilities[i] * 100, 2)

            if suitability >= 85:
                category = "Strong"
            elif suitability >= 60:
                category = "Good"
            else:
                category = "Moderate"

            data.append({
                "Resume": os.path.basename(resume_ids[idx]),
                "Similarity (%)": similarity,
                "Suitability (%)": suitability,
                "Category": category
            })

        df = pd.DataFrame(data)

        # STORE RESULTS
        st.session_state["results_df"] = df
        st.session_state["resume_count"] = len(resume_texts)
        st.session_state["time_taken"] = round(time.time() - start_time, 2)


# ============================================================
# DISPLAY (PERSISTENT)
# ============================================================

if "results_df" in st.session_state:

    df = st.session_state["results_df"]

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Total Resumes", st.session_state["resume_count"])
    col2.metric("🎯 Top Match", f"{df['Similarity (%)'].max()}%")
    col3.metric("🔥 Strong Matches", len(df[df["Category"] == "Strong"]))
    col4.metric("⚡ Time", f"{st.session_state['time_taken']}s")

    st.markdown("---")

    # FILTER + SEARCH
    st.subheader("🔍 Filter & Search")

    col1, col2 = st.columns(2)

    with col1:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All", "Strong", "Good", "Moderate"],
            key="category_filter"
        )

    with col2:
        search = st.text_input(
            "Search Resume",
            key="search_input"
        )

    # APPLY FILTER
    filtered_df = df.copy()

    if category_filter and category_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category_filter
        ]

    # Apply search
    if search and search.strip() != "":
        filtered_df = filtered_df[
            filtered_df["Resume"].str.contains(search, case=False, na=False)
        ]

    # TABLE
    st.subheader("📋 Ranked Resumes")
    st.dataframe(filtered_df, use_container_width=True)

    # CHARTS
    st.subheader("📊 Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(filtered_df.set_index("Resume")["Similarity (%)"])

    with col2:
        st.bar_chart(filtered_df["Category"].value_counts())

    st.success("✅ Matching Complete!")