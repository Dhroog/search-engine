from nltk.stem import PorterStemmer

# Create a stemmer object
stemmer = PorterStemmer()


def tokenize(text):
    return text.split()


def lowercase_filter(tokens):
    return [token.lower() for token in tokens]


def stem_filter(tokens):
    return [stemmer.stem(token) for token in tokens]


import re
import string

PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))


def punctuation_filter(tokens):
    return [PUNCTUATION.sub('', token) for token in tokens]


# top 25 most common words in English and "wikipedia":
# https://en.wikipedia.org/wiki/Most_common_words_in_English
STOPWORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'wikipedia'])


def stopword_filter(tokens):
    return [token for token in tokens if token not in STOPWORDS]


def analyze(text):
    if isinstance(text, float):
        text = str(text)
    tokens = tokenize(text)
    tokens = lowercase_filter(tokens)
    tokens = punctuation_filter(tokens)
    tokens = stopword_filter(tokens)
    # tokens = stem_filter(tokens)

    return [token for token in tokens if token]
