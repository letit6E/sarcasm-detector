import dill
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from all_data.get_path import get_file_path
from data_tokenization.data_tokenizer import my_tokenizer


def create_tfidf_vectorizer():
    tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer,
                                       # наш токенайзер (твит токенайзер + кастомное приведение к регистру)
                                       stop_words=stopwords.words('english'),  # стоп-слова
                                       ngram_range=(1, 2),  # n-граммы
                                       min_df=2,  # игнорим слова, которые встречаются редко (< 2 раз)
                                       max_df=0.95  # игнорим слова, которые встречаются часто (> 95%)
                                       )
    X_train = pd.read_csv(get_file_path('X_train.csv')).dropna()['comment'].tolist()
    tfidf_vectorizer.fit(X_train)

    # Сохранение векторизатора в файл
    tfidf_vectorizer_path = get_file_path('tfidf_vectorizer.pkl')
    with open(tfidf_vectorizer_path, 'wb') as f:
        dill.dump(tfidf_vectorizer, f)
