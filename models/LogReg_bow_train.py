import os

import dill
from all_data.get_path import get_file_path
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_tokenization.bow_vectorizer import create_bow_vectorizer


def LogReg_bow_train() -> int:
    X_train = pd.read_csv(get_file_path('X_train.csv'))
    y_train = pd.read_csv(get_file_path('y_train.csv'))
    X_train = X_train['comment'].tolist()
    y_train = y_train['label'].tolist()
    if not os.path.exists(get_file_path('bow_vectorizer.pkl')):
        create_bow_vectorizer()
    with open(get_file_path('bow_vectorizer.pkl'), 'rb') as f:
        bow_vectorizer = dill.load(f)

    X_train_bow = bow_vectorizer.transform(X_train)
    LogReg_bow = LogisticRegression(n_jobs=-1, max_iter=10000)
    LogReg_bow.fit(X_train_bow, y_train)

    LogReg_bow_model_path = get_file_path('LogReg_bow_model.pkl')
    with open(LogReg_bow_model_path, 'wb') as f:
        dill.dump(LogReg_bow, f)


