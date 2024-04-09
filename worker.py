import re
import nltk
import string
import zipfile
import os.path
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def prepare_data():
    nltk.download('stopwords')
    nltk.download('punkt')

    if not os.path.isfile('models/tfidf_vectorizer.pkl'):
        with zipfile.ZipFile('models/tfidf_vectorizer.zip', 'r') as zip_ref:
            zip_ref.extractall('models')

def preprocess(text:str):
    text = re.sub('[?!]+',' featuremark ',text)
    text = re.sub('\.{2,}',' featuredot ',text)
    text = re.sub('[0-9]+','',text)

    words = TweetTokenizer().tokenize(text)
    punct = list(string.punctuation)

    words = [word.lower() if not word.isupper() else word for word in words ]
    custom_sw = ["'s","``","'m","'d","'re","--", "(",")","'d",""," ","n't","'t","'"]
    sw = set(list(stopwords.words('english')) + punct + custom_sw)
    words = [word for word in words if word not in sw]
    preprocessed_text = ' '.join(words)

    tfidf = load('models/tfidf_vectorizer.pkl')
    return tfidf.transform([preprocessed_text])

def detect(text):
    model = load('models/LogReg.joblib')
    return model.predict(preprocess(text))
