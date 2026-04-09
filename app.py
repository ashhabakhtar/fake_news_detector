import streamlit as st
import joblib
import os
from preprocessing import clean_text

# Load model and vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

st.set_page_config(page_title="AI Fake News Detector", page_icon="🕵️")

st.title("🕵️ Fake News Detector")
st.write("Using Machine Learning to verify news authenticity.")

if os.path.exists('models/best_model.pkl'):
    model, tfidf = load_assets()
    
    user_input = st.text_area("Paste the news article text here:", height=250)
    
    if st.button("Check News"):
        if user_input.strip():
            with st.spinner('Analyzing...'):
                cleaned = clean_text(user_input)
                vec = tfidf.transform([cleaned])
                prediction = model.predict(vec)[0]
                
                # PassiveAggressive doesn't do 'predict_proba', so we show result
                st.divider()
                if prediction == 1:
                    st.success("### ✅ Result: LIKELY REAL NEWS")
                    st.balloons()
                else:
                    st.error("### 🚨 Result: LIKELY FAKE / MISLEADING")
        else:
            st.warning("Please paste some text first.")
else:
    st.error("Model files not found! Please run 'python train.py' first.")