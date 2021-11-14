import os
import re

import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
import torch
import nltk
from nltk.corpus import stopwords

import preprocessor as p
import time
from sklearn.metrics import accuracy_score, classification_report
import logging


# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class BertClassifier(object):
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.configs = configs
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        self.stopwords = set(stopwords.words('english'))
        self.contractions = pd.read_json(self.configs["contraction_file"], typ='series')
        self.contractions = self.contractions.to_dict()
        self.c_re = re.compile('(%s)' % '|'.join(self.contractions.keys()))

    def transformers(self, text):
        des_embeddings = []
        for i, des in enumerate(text):
            des_embeddings.append(self.model.encode(des))
        return des_embeddings

    def replace(self, match):
        return self.contractions[match.group(0)]

    def expandContractions(self, text):
        return self.c_re.sub(self.replace, text)

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

        return cleaned_dataset

    def load(self):
        self.model = ort.InferenceSession(self.configs["model_onnx"])

        # self.model = DistilBertForSequenceClassification.from_pretrained(self.configs["model_path"])
        # self.model.eval()
        # self.model.to("cpu")

        logging.info("DistilBertForSequenceClassification Load Sucessfully!")

    def train(self, train_file, test_file):
        self.train_dataset = pd.read_csv(train_file)
        self.train_dataset = self.train_dataset.sample(frac=1.0)
        self.test_dataset = pd.read_csv(test_file)
        self.test_dataset = self.test_dataset.sample(frac=1.0)

        train_x = self.clean(self.train_dataset['text'])
        test_x = self.clean(self.test_dataset['text'])

        # print(cleaned_dataset)

        self.x_train = self.transformers(train_x)
        self.x_test = self.transformers(test_x)

        self.y_train = self.train_dataset.label.tolist()
        self.y_test = self.test_dataset.label.tolist()
        # print(self.x_train)
        self.sgd.fit(self.x_train, self.y_train)

    def evaluate(self):
        y_pred = self.sgd.predict(self.x_test)
        print('accuracy %s' % accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred, digits=5))

    def process_one(self, input):
        s1 = time.time()
        text = input.lower()
        text = self.BAD_SYMBOLS_RE.sub(' ', text)
        text = p.clean(text)
        # expand contraction
        text = self.expandContractions(text)
        # remove punctuation
        text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())
        # stop words
        word_tokens = nltk.word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if w not in self.stopwords]
        text = ' '.join(filtered_sentence)

        logging.info(f"preprocessed text: {text}")
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=100, return_tensors='pt')
        logging.info(f"text preprocessing time: {time.time() - s1} ")
        s2 = time.time()

        input_feed = {'input_ids': encodings.input_ids.numpy(),
                      'attention_mask': encodings.attention_mask.numpy()}
        logits = self.model.run(['output'], input_feed)[0][0]
        # logits = self.model(**self.tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=100).to("cpu")).logits[0].detach().cpu().numpy()
        probs = softmax(logits)
        logging.info(f" text inference time: {time.time() - s2} ")
        logging.info(f" text predictions depression prob: {probs[1]} ")
        return {'depression': probs[1], 'nondepression': probs[0]}
