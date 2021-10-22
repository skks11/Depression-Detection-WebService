import contractions
import numpy as np
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
import xgb as xgb
import nltk
from nltk.corpus import stopwords

import preprocessor as p

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier


class TextModel(object):
    def __init__(self):
        self.model = None
        self.X = []  # X-- clean dataset
        self.sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5,
                                 tol=None)

    def transformers(self, text, detail):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        text = detail['content'].tolist()
        des_embeddings = []
        for i, des in enumerate(text):
            des_embeddings.append(self.model.encode(des))
        # des_embeddings = des_embeddings.dropna()
        # print(des_embeddings)
        self.sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5,
                                 tol=None)
        return des_embeddings

    def expandContractions(self, clean_path, text):
        contractions = pd.read_json(os.path.join(clean_path), typ='series')
        contractions = contractions.to_dict()
        c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

        def replace(match):
            return contractions[match.group(0)]

        return c_re.sub(replace, text)

    def clean(self, info, clean_path):
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        cleaned_dataset = []
        texts = [text for text in info]
        nltk.download('stopwords')
        nltk.download('punkt')
        for text in texts:
            text = str(text)
            text = text.lower()
            text = BAD_SYMBOLS_RE.sub(' ', text)
            text = p.clean(text)
            # expand contraction
            text = self.expandContractions(clean_path, text)
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

    def process_one(self, data_path, clean_path):
        encoding = 'ISO-8859-1'
        col_names = ['id', 'content', 'label']
        dataset = pd.read_csv(os.path.join(data_path), encoding=encoding, names=col_names)
        self.X = self.clean(dataset['content'], clean_path)
        self.X = self.transformers(self.X, dataset)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, dataset.label,
                                                                                test_size=0.3, random_state=42)
        self.sgd.fit(self.x_train, self.y_train)
        y_pred = self.sgd.predict(self.x_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred, digits=5))


text_model = TextModel()
# text_model.clean()
# text_model.train()
# text_model.pred()
text_model.process_one('./input/final.csv', './input/contractions.json')
