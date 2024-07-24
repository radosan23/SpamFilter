import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string


def preprocess(text, model):
    text = text.lower()
    tokens = [i.lemma_ for i in model(text)]
    tokens = [i.translate(str.maketrans('', '', string.punctuation)) for i in tokens]
    tokens = ['aanumbers' if re.search(r'\d+', i) else i for i in tokens]
    tokens = [i for i in tokens if (len(i) > 1) & (i not in STOP_WORDS)]
    return ' '.join(tokens)


def main():
    df = pd.read_csv('../data/spam.csv', header=0, names=['Target', 'SMS'], usecols=[0, 1],
                     encoding='iso-8859-1')
    pd.options.display.max_columns = df.shape[1]
    pd.options.display.max_rows = df.shape[0]

    model = spacy.load('en_core_web_sm')
    df.SMS = df.SMS.apply(preprocess, args=(model,))
    print(df.head(200))


if __name__ == '__main__':
    main()
