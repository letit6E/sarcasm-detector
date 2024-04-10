import pandas
import re
import string
import pandas as pd


def drop_objects(path: str) -> str:
    df = pd.read_csv(path)
    df.drop(['author', 'date', 'created_utc', 'parent_comment', 'subreddit', 'downs', 'ups', 'score'], axis=1,
            inplace=True)  # дропаем лишние колонки
    df.dropna(inplace=True)  # дропаем nan
    df = df[df['comment'].str.len() >= 5]  # оставляем комменты с длиной не меньше 5 символов
    df = df[df['comment'].apply(lambda x: len(x.split()) <= 15)]  # слишком длинные тоже дропаем
    df.reset_index().drop('index', axis=1, inplace=True)  # ресетаем и дропаем индексы
    df.to_csv(path, index=False)
    return path


def clean_text(text: str) -> str:
    text = re.sub('[?!]+', ' featuremark ', text)  # признак с ?!?!??
    text = re.sub('\.{2,}', ' featuredot ', text)  # многоточие
    text = re.sub('[0-9]+', '', text)  # удаляем цифры
    text = re.sub(r'[^\w\s\?\!\(\):]', '',
                  text)  # удаляем спец. символы кроме скобок, двоеточия, знаков вопроса и восклицания
    text = re.sub(r'\s+', ' ', text)  # удаляем лишние пробелы
    return text


def clead_dataframe(path: str) -> str:
    df = pd.read_csv(path)
    df_cleaned = df.copy()
    df_cleaned['comment'] = df_cleaned['comment'].apply(lambda text: clean_text(text))
    path = 'sarcasm-detector/clean_dataset.csv'
    df_cleaned.to_csv(path, index=False)
    return path
