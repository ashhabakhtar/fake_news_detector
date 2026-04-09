import pandas as pd
import numpy as np
import joblib
import os
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text, extract_style_features

def train_smart_model():
    print("🚀 Step 1: Loading Kaggle Datasets...")
    try:
        # Loading the real datasets from your 'data' folder
        fake = pd.read_csv('data/Fake.csv')
        true = pd.read_csv('data/True.csv')
    except FileNotFoundError:
        print("❌ Error: data/Fake.csv or data/True.csv not found!")
        return

    # Labeling: 0 for Fake, 1 for Real
    fake['label'], true['label'] = 0, 1
    
    # Combine and shuffle the data
    df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
    
    # We use 20,000 samples to ensure training is fast but accurate
    df = df.head(20000) 

    print("🧠 Step 2: Extracting Text Features (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    # Combine Title and Text for maximum context
    X_tfidf = tfidf.fit_transform(df['title'] + " " + df['text'].apply(clean_text))

    print("📊 Step 3: Extracting Style Features (Sentiment & Tone)...")
    # This calls the extract_style_features function from your preprocessing.py
    X_style = np.array([extract_style_features(t) for t in df['text']])

    print("🔗 Step 4: Merging Hybrid Features...")
    # Horizontally stack the text matrix and the style features array
    X_final = sp.hstack([X_tfidf, X_style])
    y = df['label']

    # Splitting for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    print("🌲 Step 5: Training Smart Random Forest Classifier...")
    # n_jobs=-1 uses all your computer's CPU cores to speed things up
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    print(f"\n✅ Training Complete! Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print("\n--- Detailed Performance Report ---")
    print(classification_report(y_test, preds))

    # --- THE SAVING PART ---
    # This creates the 'models' folder if it doesn't exist
    if not os.path.exists('models'): 
        os.makedirs('models')
    
    # Save the model and the vectorizer so the app can use them later
    joblib.dump(model, 'models/smart_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("\n💾 SUCCESS: Model files saved in /models folder!")

# This ensures the script only runs when executed directly
if __name__ == "__main__":
    train_smart_model()