import pandas as pd
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import pickle

class PersonalAttributeModel(object):
    def __init__(self):
        self.xgb = None
        
    def feature_extract():
        """
        extract feature from raw data
        """
        pass

    def train(self, data_path='./src/depressed_dataset.csv'):
        self.df = pd.read_csv(data_path)
        self.dfDrop = self.df.drop(['no_lasting_investmen', 'Survey_id', 'Ville_id', 'gained_asset', 'durable_asset', 'save_asset', 'farm_expenses', 'labor_primary', 'Number_children','lasting_investment','incoming_agricultural','incoming_own_farm' , 'incoming_business' , 'incoming_no_business'], axis=1)
        x_1 = self.dfDrop[self.dfDrop.depressed ==1]
        x_0 = self.dfDrop[self.dfDrop.depressed ==0]
        new_df = pd.concat([x_1, x_1, x_1, x_1 , x_1, x_0])
        X = new_df.iloc[:, :-1].values
        y = new_df.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2)
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label= self.y_test)
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'gamma': 0.1,
            'max_depth': 8,
            'alpha': 0,
            'lambda': 0,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'min_child_weight': 3,
            'eta': 0.03,
            'nthread': -1,
            'seed': 2019,
        }
        num_boost_round=500
        self.xgb = xgb.train(params, dtrain, num_boost_round, verbose_eval=200)


    def test(self):
        dtest = xgb.DMatrix(self.X_test, label= self.y_test)
        preds = self.xgb.predict(dtest)

        preds = np.array(preds)
        preds[preds>0.5]=1
        preds[preds<=0.5]=0
        # print(preds)
        print(confusion_matrix(self.y_test, preds))
        print (classification_report(self.y_test, preds))
        print("Accuracy:", accuracy_score(self.y_test, preds))
        

    def save(self):
        """
        save model
        """
        pickle.dump(self.xgb, open("xgb.dat", "wb"))

    def load(self):
        """
        load model
        """
        self.xgb = pickle.load(open("xgb.dat", 'rb'), encoding='bytes')
        # self.train()
        

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
        lis = np.array([list(map(int, inputdata.split(',')))])
        test_case = xgb.DMatrix(lis)
        return {'depression':self.xgb.predict(test_case)[0],'nondepression':1-self.xgb.predict(test_case)[0]}

   


attribute_model = PersonalAttributeModel()
attribute_model.train()
attribute_model.test()
attribute_model.save()
attribute_model.load()
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

output:
[ Zero: No depressed] or [One: depressed]
'''
# test_case
test_case = '1,32, 1, 8, 7, 15334717,52370258, 0'
test_case1 = '1,26, 1, 8, 5,33365355,13789233, 0'
print(attribute_model.process_one(test_case))
print(attribute_model.process_one(test_case1))