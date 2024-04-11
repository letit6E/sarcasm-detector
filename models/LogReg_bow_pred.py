import dill
import os
from all_data.get_path import get_file_path
from models.LogReg_bow_train import LogReg_bow_train


def LogReg_bow_pred(text: str) -> bool:
    if not os.path.exists(get_file_path('LogReg_bow_model.pkl')):
        LogReg_bow_train()
    with open(get_file_path('bow_vectorizer.pkl'), 'rb') as f:
        bow_vectorizer_from_dill = dill.load(f)
    text = bow_vectorizer_from_dill.transform([text])
    with open(get_file_path('LogReg_bow_model.pkl'), 'rb') as f:
        LogReg_bow = dill.load(f)
    return LogReg_bow.predict(text)
