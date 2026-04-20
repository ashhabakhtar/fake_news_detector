# 🕵️‍♂️ AI Smart News Verifier - Fake News Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

## 🚀 Overview
**AI Smart News Verifier** is an advanced fake news detection system that combines **semantic content analysis** (TF-IDF keywords) with **stylometry** (writing style analysis) for superior accuracy. Unlike basic keyword detectors, this hybrid approach catches sophisticated misinformation by analyzing both **what** is said and **how** it's written.

**Key Innovation**: Detects sensationalism through subjectivity, emotional polarity, punctuation density, and capitalization patterns common in clickbait/fake news.

Trained on Kaggle's Fake/True news datasets (~20k samples for fast training).

## ✨ Features
- **Hybrid ML Model**: TF-IDF + Style Features → RandomForestClassifier
- **Interactive Web UI**: Streamlit app with real-time trust scores
- **Style Analysis**: Subjectivity, sentiment polarity, punctuation/caps ratios
- **Production Ready**: Pre-trained models, one-click deployment
- **Fast Training**: Uses 20k samples, all CPU cores

## 📊 Live Demo
```
Paste article → Get Trust Score + Style Breakdown
✅ LIKELY REAL: 92% confidence
🚨 LIKELY FAKE: 87% deception risk
```

## 🛠 Quick Start

### 1. Clone & Setup
```bash
cd c:/Users/Lenovo pc/Desktop/fake_news_detector
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Train Model (First Time Only)
```bash
python train.py
```
```
🚀 Loading datasets...
✅ Accuracy: 98.5%
Models saved to /models/
```

### 3. Launch Web Server
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser! The frontend is powered by a modern, responsive HTML/CSS/JS interface communicating with our Flask API.

## 🔍 How to Use
1. Paste full news article (title + body)
2. Click **🔍 Analyze Authenticity**
3. Get instant **Trust Score** + **Style Indicators**

**Example Fake News Input**:
```
BREAKING: SHOCKING DISCOVERY! Scientists confirm Earth is FLAT! NASA hiding truth!
```
**Output**: 🚨 **LIKELY FAKE** (95% risk) - High subjectivity (0.78), extreme polarity

## 🧠 Model Architecture

```
Raw Text → [clean_text()] → TF-IDF (5000 features)
         → [extract_style_features()] → [polarity, subjectivity, punc_density, caps_ratio]
Final Features (sparse hstack) → RandomForest → Prediction + Probabilities
```

**Style Features Explained**:
- **Polarity**: Emotional charge (-1 negative, +1 positive)
- **Subjectivity**: Opinion vs fact (0=objective, 1=opinion)
- **Punctuation Density**: `!` + `?` usage (sensationalism indicator)
- **Caps Ratio**: SHOUTING/CLICKBAIT detector

## 📁 Project Structure
```
fake_news_detector/
├── app.py              # Streamlit web interface
├── train.py            # Model training + evaluation
├── preprocessing.py    # NLP cleaning + style extraction
├── requirements.txt    # Dependencies
├── data/               # Kaggle datasets (Fake.csv, True.csv)
├── models/             # smart_model.pkl, tfidf_vectorizer.pkl
├── README.md          ← You're here!
└── TODO.md            # Task progress
```

## ⚙️ Requirements
```txt
pandas numpy scikit-learn nltk streamlit joblib textblob plotly
```

## 🔬 Training Details
- **Dataset**: 10k Fake + 10k True news articles
- **Features**: 5000 TF-IDF n-grams + 4 style metrics
- **Model**: RandomForest (100 trees, max_depth=20)
- **Accuracy**: ~98% on holdout set
- **Training Time**: <5 minutes on standard laptop

## 🧪 Testing Your Model
Add to `train.py` end:
```python
# Test with new article
test_text = "Your news article here"
features = prepare_features(test_text)  # Implement based on pipeline
prediction = model.predict(features)
print("Prediction:", "REAL" if prediction else "FAKE")
```

## 🚀 Future Improvements
- [ ] BERT/Transformers for contextual embeddings
- [ ] Fact-checking API integration
- [ ] Multi-language support
- [ ] Docker deployment
- [ ] Model versioning (MLflow)

## 📝 License
MIT License - Use for good, not evil! 😇

## 🙌 Acknowledgments
- Kaggle Fake News Dataset
- Streamlit for amazing UI
- Scikit-learn & NLTK for ML/NLP

**Made with ❤️ for truth-seekers everywhere**

---
*\"The ultimate measure of a news article is not what it says, but how it says it.\"*

## 🚀 Advanced Features (v2.0)
- **Hybrid Detection:** Combines TF-IDF keyword analysis with Stylometric metadata.
- **Sentiment Analysis:** Uses `TextBlob` to detect emotional polarity and subjectivity.
- **Ensemble Learning:** Implements a **Random Forest Classifier** for higher accuracy against "nearly real" fake news.
- **Interactive Dashboard:** Advanced Streamlit UI featuring Trust Scores, Deception Risk metrics, and Style Breakdowns.


