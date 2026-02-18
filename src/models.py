import numpy as np
import gensim.downloader as api
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB

def create_baseline_pipeline(vec_type="tfidf", ngram_range=(1, 1), max_features=20000, analyzer='word'):
    """
    Creates a standard pipeline with a Vectorizer and Logistic Regression.
    Supports both 'word' and 'char' analyzers.
    """
    # 1. Choose Vectorizer
    if vec_type.lower() == "tfidf":
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=3, max_features=max_features, analyzer=analyzer)
    elif vec_type.lower() == "count":
        vec = CountVectorizer(ngram_range=ngram_range, min_df=3, max_features=max_features, analyzer=analyzer)
    else:
        raise ValueError("vec_type must be 'tfidf' or 'count'")
        
    # 2. Return Pipeline
    # Note: We use class_weight='balanced' to handle class imbalance
    return Pipeline([
        ('vec', vec),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', 
                                                       class_weight='balanced', 
                                                       random_state=42)))
    ])

def create_advanced_pipeline(model_type="logreg", vec_type="tfidf", ngram_range=(1, 1)):
    """
    Creates a pipeline with a specific classifier (LogReg, SVM, NB) for Model Selection.
    """
    # 1. Vectorizer (Same as baseline)
    if vec_type == "tfidf":
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=3, max_features=20000)
    else:
        vec = CountVectorizer(ngram_range=ngram_range, min_df=3, max_features=20000)

    # 2. Classifier Selection
    # We wrap everything in OneVsRest because this is Multi-label
    if model_type == 'logreg':
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    
    elif model_type == 'svm':
        # LinearSVC is faster and better for text than standard SVC
        clf = LinearSVC(class_weight='balanced', random_state=42, dual=False) 
        
    elif model_type == 'nb':
        # MultinomialNB is standard for text
        clf = MultinomialNB(alpha=0.1) 
        
    elif model_type == 'complement_nb':
        # ComplementNB is better for imbalanced datasets
        clf = ComplementNB(alpha=0.1)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([
        ('vec', vec),
        ('clf', OneVsRestClassifier(clf))
    ])

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Dense Embeddings (Word2Vec/GloVe).
    Averages word vectors for a document.
    """
    def __init__(self, model_name=None, word2vec=None):
        self.model_name = model_name
        self.word2vec = word2vec
        self.vector_size = None

    def fit(self, X, y=None):
        # Allow passing a pre-loaded model to save RAM
        if self.word2vec is None and self.model_name:
            print(f"Loading {self.model_name}...")
            self.word2vec = api.load(self.model_name)
        
        if self.word2vec is not None:
             # Gensim 4.0+ uses .vector_size, older versions use .vector_size
            self.vector_size = getattr(self.word2vec, 'vector_size', 300)
            
        return self

    def transform(self, X):
        if self.word2vec is None:
            raise ValueError("Model not loaded. Please provide model_name or pre-loaded model.")
            
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.vector_size)], axis=0)
            for words in [s.lower().split() for s in X]
        ])