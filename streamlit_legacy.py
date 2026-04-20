import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import os
from preprocessing import clean_text, extract_style_features

# 1. Page Configuration
st.set_page_config(
    page_title="AI Smart News Verifier",
    page_icon="🕵️",
    layout="wide"
)

# 2. Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load model and vectorizer
@st.cache_resource
def load_assets():
    model_path = 'models/smart_model.pkl'
    tfidf_path = 'models/tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf
    return None, None

# --- UI Header ---
st.title("🕵️ AI Smart News Verifier")
st.markdown("This dashboard uses a **Hybrid Machine Learning model** to analyze the factual keywords and the emotional tone of news articles.")
st.divider()

model, tfidf = load_assets()

if model is None:
    st.error("⚠️ **Model files not found!** Please run `python train.py` first to generate your smart model.")
else:
    # Sidebar for project info
    with st.sidebar:
        st.header("Project Overview")
        st.write("This detector looks for **'Deception Cues'** such as high subjectivity, sensationalist punctuation, and biased sentiment.")
        st.divider()
        st.caption("Developed as a Portfolio Project")

    # Layout: 2 Columns
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.subheader("📝 Input Article Content")
        user_input = st.text_area(
            "Paste the news text you want to verify below:",
            height=350,
            placeholder="E.g., The city council announced a new budget today... OR Breaking news! You won't believe..."
        )
        
        analyze_btn = st.button("🚀 Analyze Authenticity", use_container_width=True)

    with col2:
        st.subheader("📊 Analysis Report")
        
        if analyze_btn and user_input:
            with st.spinner('Deep scanning article structure and sentiment...'):
                # Step 1: Preprocess content
                cleaned_text = clean_text(user_input)
                
                # Step 2: Extract style features (Metadata)
                style_features = extract_style_features(user_input)
                
                # Step 3: Vectorize text
                text_vector = tfidf.transform([cleaned_text])
                
                # Step 4: Combine features for the Hybrid model
                final_features = sp.hstack([text_vector, np.array([style_features])])
                
                # Step 5: Prediction
                prediction = model.predict(final_features)[0]
                probabilities = model.predict_proba(final_features)[0]
                confidence = probabilities[prediction] * 100

                # --- Display Results ---
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.success("### Verdict: LIKELY REAL NEWS")
                    st.metric("Trust Score", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
                    st.balloons()
                else:
                    st.error("### Verdict: LIKELY FAKE / MISLEADING")
                    st.metric("Deception Risk", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.divider()
                
                # Writing Style Indicators
                st.write("**Writing Style Breakdown:**")
                
                # Display Subjectivity Score
                sub_score = style_features[1]
                st.write(f"🔹 **Subjectivity:** {sub_score:.2f}")
                if sub_score > 0.5:
                    st.caption("🚩 High subjectivity: This article sounds more like an opinion than a report.")
                else:
                    st.caption("✅ Low subjectivity: The writing appears neutral and objective.")

                # Display Sentiment Polarity
                pol_score = style_features[0]
                st.write(f"🔹 **Sentiment Tone:** {pol_score:.2f}")
                st.caption("(-1 = Highly Negative | 0 = Neutral | 1 = Highly Positive)")

        elif not analyze_btn:
            st.info("Paste an article in the text box and click 'Analyze' to see the AI verdict.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and provides an estimate based on linguistic patterns. Always check multiple sources.")