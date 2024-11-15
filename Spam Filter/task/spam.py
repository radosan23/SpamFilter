import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import numpy as np


def preprocess(text, model):
    text = text.lower()
    tokens = [i.lemma_ for i in model(text)]
    tokens = [i.translate(str.maketrans('', '', string.punctuation)) for i in tokens]
    tokens = ['aanumbers' if re.search(r'\d+', i) else i for i in tokens]
    tokens = [i for i in tokens if (len(i) > 1) & (i not in STOP_WORDS)]
    return ' '.join(tokens)


def train_test_split(df, train_ratio=0.8, random=None):
    df = df.sample(frac=1, random_state=random, ignore_index=False)
    train_last_index = int(df.shape[0] * train_ratio)
    return df.iloc[:train_last_index], df.iloc[train_last_index:]


def make_vocabulary(df):
    result = set()
    for text in df['SMS']:
        result.update(text.split())
    return sorted(result)


def bag_of_words(df, vocabulary):
    new_cols = []
    for token in vocabulary:
        new_cols.append(pd.Series(df['SMS'].apply(str.count, args=(token,)), name=token, dtype=int, index=df.index))
    return pd.concat([df, *new_cols], axis=1)


def ham_spam_prob(df, vocabulary, alpha=1):
    n_voc = len(vocabulary)
    spam_words = ' '.join(df[df['Target'] == 'spam']['SMS'].tolist()).split()
    ham_words = ' '.join(df[df['Target'] == 'ham']['SMS'].tolist()).split()
    spam_prob = [(spam_words.count(x) + alpha) / (len(spam_words) + alpha * n_voc) for x in vocabulary]
    ham_prob = [(ham_words.count(x) + alpha) / (len(ham_words) + alpha * n_voc) for x in vocabulary]
    df_nb = pd.DataFrame({'Spam Probability': spam_prob, 'Ham Probability': ham_prob}, index=vocabulary)
    return df_nb


def naive_bayes(sms, prob, p_spam, p_ham):
    p_spam_sms = p_spam * np.prod([prob.loc[word, 'Spam Probability'] for word in sms.split() if word in prob.index])
    p_ham_sms = p_ham * np.prod([prob.loc[word, 'Ham Probability'] for word in sms.split() if word in prob.index])
    if p_spam_sms > p_ham_sms:
        return 'spam'
    elif p_ham_sms > p_spam_sms:
        return 'ham'
    else:
        return 'unknown'


def calc_metrics(pred):
    pred['val'] = 1
    conf_mat = pd.pivot_table(pred, values=['val'], index=['Target'], columns=['Predicted'], aggfunc='sum')
    accuracy = ((conf_mat.iloc[1, 1] + conf_mat.iloc[0, 0]) /
                (conf_mat.iloc[1, 1] + conf_mat.iloc[0, 0] + conf_mat.iloc[0, 1] + conf_mat.iloc[1, 0]))
    recall = (conf_mat.iloc[1, 1]) / (conf_mat.iloc[1, 1] + conf_mat.iloc[1, 0])
    precision = (conf_mat.iloc[1, 1]) / (conf_mat.iloc[1, 1] + conf_mat.iloc[0, 1])
    f1 = (2 * precision * recall) / (precision + recall)
    return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}


def main():
    df = pd.read_csv('../data/spam.csv', header=0, names=['Target', 'SMS'], usecols=[0, 1],
                     encoding='iso-8859-1')

    model = spacy.load('en_core_web_sm')
    df.SMS = df.SMS.apply(preprocess, args=(model,))
    df_train, df_test = train_test_split(df, train_ratio=0.8, random=43)
    vocabulary = make_vocabulary(df_train)
    p_spam = df_train[df_train['Target'] == 'spam'].shape[0] / df_train.shape[0]
    p_ham = df_train[df_train['Target'] == 'ham'].shape[0] / df_train.shape[0]
    df_train_prob = ham_spam_prob(df_train, vocabulary=vocabulary)
    df_test['Predicted'] = df_test['SMS'].apply(naive_bayes, args=(df_train_prob, p_spam, p_ham))
    metrics = calc_metrics(df_test)

    pd.options.display.max_columns = df_test.shape[1]
    pd.options.display.max_rows = df_test.shape[0]
    print(metrics)


if __name__ == '__main__':
    main()
