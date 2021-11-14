import time

import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import wave
import librosa
import numpy as np
import xgboost as xgb
import pickle
import logging
import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D




class AudioCNN(object):
    def __init__(self, configs):
        self.imwidth = 938
        self.imheight = 80
        self.model = None
        self.configs = configs


    def feature_extract(self, wave_file):
        """
        extract feature from raw data
        """
        try:
            wavefile = wave.open(wave_file)
            sr = wavefile.getframerate()
            secs = 938 / (sr / 512)
            nframes = wavefile.getnframes()
            wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
            signal = wave_data.astype(np.float)
            clip = math.ceil(sr * secs)
            audio_feat = librosa.feature.melspectrogram(signal[:clip], n_mels=80, sr=sr)
            audio_feat = audio_feat[:, :938]
            audio_feat = (audio_feat - audio_feat.min()) / (audio_feat.max() - audio_feat.min())
            assert audio_feat.shape == (80, 938)
            return audio_feat

        except Exception as exec:
            logging.error(f"AUDIO_FEATURE_PROCESSING_ERROR|{str(exec)}")
            raise IOError(f"AUDIO_FEATURE_PROCESSING_ERROR|{str(exec)}")

        pass

    def train(self, data_path='./src/depressed_dataset.csv'):
        pass

    def test(self):
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        preds = self.xgb.predict(dtest)

        preds = np.array(preds)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        # print(preds)
        print(confusion_matrix(self.y_test, preds))
        print(classification_report(self.y_test, preds))
        print("Accuracy:", accuracy_score(self.y_test, preds))

    def save(self):
        """
        save model
        """
        pass

    def load(self):
        """
        load model
        """
        num_classes = 2
        input_shape = (self.imheight, self.imwidth, 1)
        self.model = Sequential()
        self.model.add(Conv2D(8, kernel_size=(1, 5), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, kernel_size=(1, 5), activation='relu', ))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(1, 5), activation='relu', ))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        self.model.load_weights(self.configs["model_path"])
        logging.info("AudioCNN Load Sucessfully!")


    def pred(self, feature):
        """
        return predictions: 0-1
        """
        pass

    def convert2onnx(self):
        """
        if inference time(except video model) is greater than 10ms on your machine:
            1. reduce model size
            2. use onnx https://github.com/onnx/tutorials
        """
        pass

    def process_one(self, wave_file):

        """
        input: raw data
        output: prediction
        """
        s = time.time()
        audio_feat = self.feature_extract(wave_file=wave_file)
        audio_feat = audio_feat.reshape(self.imheight, self.imwidth, 1)
        res = self.model.predict(np.array([audio_feat]))
        logging.info(f"AudioCNN inference time {time.time() - s}")
        return {'depression': res[0][1], 'nondepression': res[0][0]}

