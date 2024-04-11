import re
import pandas as pd
from sklearn.model_selection import train_test_split

from all_data.get_path import get_file_path


def drop_objects(path: str) -> str:
    df = pd.read_csv(path)
    df.drop(['author', 'date', 'created_utc', 'parent_comment', 'subreddit', 'downs', 'ups', 'score'], axis=1,
            inplace=True)  # дропаем лишние колонки
    df = df[df['comment'].str.len() >= 5]  # оставляем комменты с длиной не меньше 5 символов
    df = df[df['comment'].apply(lambda x: len(x.split()) <= 15)]  # слишком длинные тоже дропаем
    df = df[(df['label'] == 0) | (df['label'] == 1)]
    df.dropna(axis=0, inplace=True)  # дропаем nan
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
    df_cleaned.dropna(inplace=True)  # дропаем nan
    # df_cleaned.reset_index().drop('index', axis=1, inplace=True)
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
    df_cleaned_path = get_file_path(call_clean(path))
    df_cleaned = pd.read_csv(df_cleaned_path)
    df_cleaned.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned['comment'],
                                                        df_cleaned['label'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_cleaned['label']
                                                        )

    X_train = {'comment': X_train.tolist()}
    X_test = {'comment': X_test.tolist()}
    y_train = {'label': y_train.tolist()}
    y_test = {'label': y_test.tolist()}

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    X_train_path = get_file_path('X_train.csv')
    X_test_path = get_file_path('X_test.csv')
    y_train_path = get_file_path('y_train.csv')
    y_test_path = get_file_path('y_test.csv')

    X_train.to_csv(X_train_path)
    X_test.to_csv(X_test_path)
    y_train.to_csv(y_train_path)
    y_test.to_csv(y_test_path)


make_train_test('sarcasm_dataset.csv')
