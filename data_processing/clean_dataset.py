import re
import pandas as pd
from sklearn.model_selection import train_test_split

from all_data.get_path import get_file_path


def drop_objects(path: str) -> str:
    df = pd.read_csv(path)
    df.drop(['author', 'date', 'created_utc', 'parent_comment', 'subreddit', 'downs', 'ups', 'score'], axis=1,
            inplace=True)  # дропаем лишние колонки
    df.dropna(inplace=True)  # дропаем nan
    df = df[df['comment'].str.len() >= 5]  # оставляем комменты с длиной не меньше 5 символов
    df = df[df['comment'].apply(lambda x: len(x.split()) <= 15)]  # слишком длинные тоже дропаем
    df.reset_index().drop('index', axis=1, inplace=True)  # ресетаем и дропаем индексы
    new_path = "df_dropped.csv"
    df.to_csv(get_file_path(new_path), index=False)
    return new_path


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
    path = get_file_path('clean_dataset.csv')
    df_cleaned.to_csv(path, index=False)
    return path


def call_clean(path: str) -> str:
    # path = 'sarcasm_dataset.csv'
    df_path = get_file_path(path)
    df_dropped_path = drop_objects(df_path)
    df_cleaned_path = clead_dataframe(get_file_path(df_dropped_path))
    return df_cleaned_path


def make_train_test(path: str):
    # path = 'sarcasm_dataset.csv'
    df_cleaned_path = get_file_path(call_clean(path))
    df_cleaned = pd.read_csv(df_cleaned_path)
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned['comment'],
                                                        df_cleaned['label'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_cleaned['label']
                                                        )
    X_train.to_csv(get_file_path('X_train.csv'), index=0)
    X_test.to_csv(get_file_path('X_test.csv'), index=0)
    y_train.to_csv(get_file_path('y_train.csv'), index=0)
    y_test.to_csv(get_file_path('y_test.csv'), index=0)


make_train_test('sarcasm_dataset.csv')
