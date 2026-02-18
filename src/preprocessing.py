import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Ensure resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def create_text_features(df):
    """
    Combines Conclusion, Stance, and Premise into a single 'text' column.
    """
    df = df.copy()
    
    # We ensure columns exist and fill NaNs to prevent errors
    for col in ['Conclusion', 'Stance', 'Premise']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from DataFrame")
        df[col] = df[col].fillna("")
        
    # Concatenate
    df['text'] = df['Conclusion'] + " " + df['Stance'] + " " + df['Premise']
    return df

def get_wordnet_pos(word):
    """Map NLTK POS tag to WordNet POS tag"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text, strategy='lower'):
    """
    Applies specific preprocessing strategies.
    Strategies: 'raw', 'lower', 'no_punct', 'no_stopwords', 'lemmatized'
    """
    if pd.isna(text): return ""
    text = str(text)

    if strategy == 'raw':
        return text
        
    text = text.lower()
    
    if strategy == 'lower':
        return text
        
    if strategy == 'no_punct':
        return re.sub(r'[^\w\s]', '', text)
        
    if strategy == 'no_stopwords':
        return " ".join([w for w in text.split() if w not in stop_words])
        
    if strategy == 'lemmatized':
        # Advanced POS-aware lemmatization
        words = text.split()
        return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words])
        
    return text