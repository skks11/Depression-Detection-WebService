import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
import missingno as missing
import seaborn as sns
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score ,auc, plot_roc_curve
from sklearn import svm
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class PersonalAttributeModel(object):
    def __init__(self, data_path):
        self.model = None
        self.df = pd.read_csv(data_path)
        self.dfDrop = self.df.drop(['no_lasting_investmen', 'Survey_id', 'Ville_id', 'gained_asset', 'durable_asset', 'save_asset', 'farm_expenses', 'labor_primary', 'Number_children','lasting_investment','incoming_agricultural'], axis=1)
        X = self.dfDrop.iloc[:, :-1].values
        y = self.dfDrop.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2)
    
    def feature_extract():
        """
        extract feature from raw data
        """
        pass

    def train(self):
        # self.rf = RandomForestClassifier(n_estimators = 9,
        #                             max_depth=3,
        #                             min_samples_split=9,
        #                             min_samples_leaf=5
        #                            )
        # self.rf.fit(self.X_train, self.y_train)
        self.knn=KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.X_train, self.y_train)

    def test(self):
        # self.rf_pred = self.rf.predict(self.X_test)
        pass

    def save():
        """
        save model
        """
        pass

    def load():
        """
        load model
        """
        pass

    def pred(self, feature):
        """
        return predictions: 0-1
        """
        pass

    def convert2onnx():
        """
        if inference time(except video model) is greater than 10ms on your machine:
            1. reduce model size
            2. use onnx https://github.com/onnx/tutorials
        """
        pass
    
    def process_one(self, inputdata):
        """
        input: raw data 
        output: prediction
        """
        lis = [list(map(int, inputdata.split(',')))]
        print(self.knn.predict_proba(lis))

   

shishu_model = PersonalAttributeModel('./src/depressed_dataset.csv')
shishu_model.train()

'''
input:
sex: [0: man] [1: woman]
Age: [1-100]
Married:[0: no] [1: yes]
education_level: Years of education completed
total_members: Household size
living_expenses: year
other_expenses: year
incoming_salary:[0: no incoming salary] [1: have incoming salary]
incoming_own_farm:[0: no incoming farm] [1: have incoming farm]
incoming_business:[0: no incoming business] [1: have incoming business]
incoming_no_business:[0: no incoming flow business] [1: have incoming flow business]

output:
[ Zero: No depressed] or [One: depressed]
'''
# test_case
test_case = '1,32, 1, 8, 7, 15334717,52370258, 0, 1, 0, 1'
test_case1 = '1,26, 1, 8, 5,33365355,13789233, 0, 0, 0, 0'
shishu_model.process_one(test_case)
shishu_model.process_one(test_case1)