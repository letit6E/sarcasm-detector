from nltk import TweetTokenizer


def half_caps(word: str) -> str:
    '''
    Проверяем, что бОльшая часть слова записана капсом.
    Если как минимум половина слова записана капсом,
    то не приводим слово к нижнему регистру
    '''
    caps_count = sum(1 for letter in word if letter.isupper())
    return word.lower() if caps_count < len(word) / 2 else word


def my_tokenizer(text: str) -> list:
    '''
    Токенизируем его с помощью TweetTokenizer
    Каждый токен при необходимости приводится к нижнему регистру
    '''
    words = [half_caps(word) for word in TweetTokenizer().tokenize(text)]
    return words
