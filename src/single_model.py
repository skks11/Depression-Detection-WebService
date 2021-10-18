import contractions
import numpy as np
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords

import preprocessor as p

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier


class TextModel(object):
    def __init__(self, data_path, clean_path):
        self.model = None
        encoding = 'ISO-8859-1'
        col_names = ['id', 'content', 'label']
        self.dataset = pd.read_csv(os.path.join(data_path), encoding=encoding, names=col_names)

        self.contractions = pd.read_json(os.path.join(clean_path), typ='series')
        self.contractions = self.contractions.to_dict()
        self.c_re = re.compile('(%s)' % '|'.join(self.contractions.keys()))
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        self.X = []  # X-- clean dataset

        self.sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5,
                                 tol=None)

    def transformers(self, text):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # self.dataset["content"] = [float(str(i).replace(",", "")) for i in self.dataset["content"]]
        text = self.dataset['content'].tolist()
        des_embeddings = []
        for i, des in enumerate(text):
            des_embeddings.append(self.model.encode(des))
        # des_embeddings = des_embeddings.dropna()
        # print(des_embeddings)
        return des_embeddings

    def expandContractions(self, text):
        def replace(match):
            return contractions[match.group(0)]

        return self.c_re.sub(replace, text)

    def clean(self):
        cleaned_dataset = []
        texts = [text for text in self.dataset['content']]
        nltk.download('stopwords')
        nltk.download('punkt')
        for text in texts:
            text = str(text)
            text = text.lower()
            text = self.BAD_SYMBOLS_RE.sub(' ', text)
            text = p.clean(text)
            # expand contraction
            text = self.expandContractions(text)
            # remove punctuation
            text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())
            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(filtered_sentence)

            cleaned_dataset.append(text)
            self.X = cleaned_dataset

        # return cleaned_dataset

    def train(self):
        self.X = self.transformers(self.X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.dataset.label,
                                                                                test_size=0.3, random_state=42)
        # print(self.x_train)
        self.sgd.fit(self.x_train, self.y_train)

    def pred(self):
        y_pred = self.sgd.predict(self.x_test)
        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred, digits=5))


text_model = TextModel('./input/final.csv', './input/contractions.json')
text_model.clean()
text_model.train()
text_model.train()
text_model.pred()
