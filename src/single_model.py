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

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier


class single_model(object):
    def __init__(self, data_path):
        self.model = None
        encoding = 'ISO-8859-1'
        col_names = ['id', 'content', 'label']
        self.dataset = pd.read_csv(os.path.join(data_path), encoding=encoding, names=col_names)

        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        self.X = []  # X-- clean dataset

        self.sgd = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=5,
                                 tol=None)

    def transformers(self, text):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        des_embeddings = []
        for i, des in enumerate(text):
            des_embeddings.append(self.model.encode(des))
        return des_embeddings

    def expandContractions(self, text):
        contractions = pd.read_json(os.path.join('./input/contractions.json'), typ='series')
        contractions = contractions.to_dict()
        c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

        def replace(match):
            return contractions[match.group(0)]

        return c_re.sub(replace, text)

    def clean(self, info):
        cleaned_dataset = []
        texts = [text for text in info]
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
            # text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())
            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(filtered_sentence)

            cleaned_dataset.append(text)
            self.X = cleaned_dataset

        # return cleaned_dataset

    def train(self):
        self.clean(self.dataset['content'])

        # print(cleaned_dataset)
        self.X = self.transformers(self.X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.dataset.label,
                                                                                test_size=0.3, random_state=42)
        # print(self.x_train)
        self.sgd.fit(self.x_train, self.y_train)

    def pred(self):
        y_pred = self.sgd.predict(self.x_test)

        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred, digits=5))

    def process_one(self, input):
        text = input.lower()
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
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # text = text.tolist()
        # self.dataset["content"] = [float(str(i).replace(",", "")) for i in self.dataset["content"]]
        des_embeddings = []
        for i, des in enumerate(text):
            des_embeddings.append(self.model.encode(des))
        result = self.sgd.predict_proba(des_embeddings)
        return {'depression': result[0][0], 'nondepression': 1 - result[0][0]}


text_model = single_model('./input/final.csv')
# text_model.clean()
text_model.train()
# text_model.pred()
test_case = "Most of us feel sad, lonely, or depressed at times. It's a normal reaction to loss, life's struggles, or injured self-esteem. But when these feelings become overwhelming, cause physical symptoms, and last for long periods of time, they can keep you from leading a normal, active life."
print(text_model.process_one(test_case))
