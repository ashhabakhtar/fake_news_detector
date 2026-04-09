import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text

def load_and_train():
    print("Step 1: Loading Kaggle datasets...")
    try:
        fake = pd.read_csv('data/Fake.csv')
        true = pd.read_csv('data/True.csv')
    except FileNotFoundError:
        print("❌ Error: Could not find Fake.csv or True.csv in the 'data' folder.")
        return

    # Labeling: 0 for Fake, 1 for Real
    fake['label'] = 0
    true['label'] = 1
    
    # Merge and use title + text
    df = pd.concat([fake, true]).reset_index(drop=True)
    df['content'] = df['title'] + " " + df['text']
    
    print(f"Step 2: Cleaning {len(df)} articles... (This takes about 1-2 minutes)")
    df['content'] = df['content'].apply(clean_text)
    
    print("Step 3: Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['content'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 4: Training Passive Aggressive Classifier...")
    # This model is specifically great for large text datasets
    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(X_train, y_train)
    
    # Quick Eval
    preds = model.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    
    print("Step 5: Saving model...")
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("🚀 DONE! Everything saved in /models")

if __name__ == "__main__":
    load_and_train()