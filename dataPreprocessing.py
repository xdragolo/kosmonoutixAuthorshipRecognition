import re
import nltk
import numpy as np
import pandas
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

from stop_words import get_stop_words


def cleanData(df):
    df['content'].replace('', np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)

    # drop extremly long or short articles
    df['length'] = df['content'].str.len()
    q25 = df['length'].quantile(q=0.25)
    df = df.loc[(q25 <= df['length'])]

    # drop articles of authors who have written less than 10 articles
    nAuthors = df['author'].nunique()
    groupby = df.groupby(by='author').count().sort_values('title')
    lazyAuthors = groupby.loc[groupby['title'] < 100].index.tolist()
    df = df[~df['author'].isin(lazyAuthors)]
    nAuthors = df['author'].nunique()
    authors = df['author'].unique()

    # represent authors name with numeric value
    authorsDict = {}
    # for i in range(nAuthors):
    #     authorsDict[authors[i]] = i
    # df = df.replace({'author' : authorsDict})

    # balance class
    removedArticles = df.loc[df['author'] == 'Dušan Majer'].index.tolist()[400:]
    df.drop(removedArticles, inplace=True)

    # drop columns title date length
    df = df[['author', 'content']]
    df = df.reset_index(drop=True)

    # clean text
    df['content'] = df['content'].apply(cleanText)
    df.to_csv('./cleanData.csv', sep=';')

    # figure of  data classes
    # df.groupby(by='author').count().plot(kind='bar')
    # plt.savefig('./figures/dataClasses.png', bbox_inches='tight')
    # plt.show()

    return df, nAuthors


def cleanText(text):
    text = text.replace('\n', '')
    replacedChars = re.compile('[\.,\–\-:"\?!“„(…)]')
    stopwords = get_stop_words('czech')

    text = text.lower()
    text = replacedChars.sub('', text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords])
    return text


def vectorizeArticles(df, n_words=50000, seq_length=400):
    tokenizer = Tokenizer(num_words=n_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['content'].values)
    wordIdx = tokenizer.word_index

    X = tokenizer.texts_to_sequences(df['content'].values)
    X = pad_sequences(X, maxlen=seq_length)

    Y = pd.get_dummies(df['author'])
    return X, Y
