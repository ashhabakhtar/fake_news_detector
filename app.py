"""
app.py — Flask API server for AI News Verifier
Run with: python app.py
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import scipy.sparse as sp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text, extract_style_features

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='frontend', static_url_path='')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'smart_model.pkl')
TFIDF_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_vectorizer.pkl')

model = None
tfidf = None


def load_models():
    global model, tfidf
    if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        print("[OK]  ML models loaded successfully.")
    else:
        print("[WARN] Model files not found! Run 'python train.py' first.")

# Load models when the module is imported (required for WSGI like gunicorn)
load_models()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if model is None or tfidf is None:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 503

    payload = request.get_json(silent=True)
    if not payload or not payload.get('text', '').strip():
        return jsonify({'error': 'Please provide a news article to analyze.'}), 400

    text = payload['text'].strip()

    try:
        cleaned    = clean_text(text)
        features   = extract_style_features(text)   # [polarity, subjectivity, ...]
        vector     = tfidf.transform([cleaned])
        combined   = sp.hstack([vector, np.array([features])])
        pred       = int(model.predict(combined)[0])
        probs      = model.predict_proba(combined)[0]
        confidence = float(probs[pred] * 100)

        return jsonify({
            'is_real':      pred == 1,
            'confidence':   round(confidence, 2),
            'sentiment':    round(float(features[0]), 4),
            'subjectivity': round(float(features[1]), 4),
            'word_count':   len(text.split()),
        })
    except Exception as exc:
        return jsonify({'error': f'Analysis error: {str(exc)}'}), 500


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[START] AI News Verifier running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
