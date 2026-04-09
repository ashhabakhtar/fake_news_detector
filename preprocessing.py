import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Ensure all necessary NLP resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Performs standard NLP cleaning.
    Removes HTML tags, URLs, punctuation, and stopwords.
    """
    text = str(text).lower()
    # Remove bracketed text, URLs, and punctuation
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove words that contain numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Tokenization and Lemmatization
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(words)

def extract_style_features(text):
    """
    Extracts 'Deception Cues' from the text style.
    This is what catches 'nearly real' fake news.
    """
    text = str(text)
    blob = TextBlob(text)
    
    # 1. Sentiment Polarity: Is the text very angry or very happy? (-1 to 1)
    polarity = blob.sentiment.polarity 
    
    # 2. Subjectivity: Does it sound like an opinion? (0 to 1)
    # Fake news usually has high subjectivity.
    subjectivity = blob.sentiment.subjectivity
    
    # 3. Punctuation Density: Fake news uses more '!' and '?'
    punc_count = (text.count('!') + text.count('?')) / (len(text) + 1)
    
    # 4. Caps Ratio: High use of uppercase letters (clickbait style)
    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    
    return [polarity, subjectivity, punc_count, caps_ratio]