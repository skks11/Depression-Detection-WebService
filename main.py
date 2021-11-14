# -*- coding: utf-8 -*-
import requests
import time
import json
from typing import Dict
import logging
from flask_cors import *
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
from PersonalAttributeModel import PersonalAttributeModel
from BertClassifier import BertClassifier
from AudioCNN import AudioCNN
import filetype
app = Flask(__name__)
CORS(app, supports_credentials=True)

logging.basicConfig(level=logging.DEBUG)

# load models

# model_1
personal_attribute_model = PersonalAttributeModel()
personal_attribute_model.load()

numerical_attributes = ["Age",  "education_level", "total_members", "living_expenses", "other_expenses"]
categorical_attributes = ["sex", "Married", "incoming_salary"]

# model_2
text_configs = {
        "model_path": "./models/text/checkpoint-best",
        "model_onnx": "./models/text/distill_bert_clf.onnx",
        "contraction_file": "./models/text/contractions.json"
}
bert_clf = BertClassifier(text_configs)
bert_clf.load()


# model_3
audio_configs = {
        "model_path": "./models/audio/best_1103.hdf5"
}
audio_clf = AudioCNN(audio_configs)
audio_clf.load()



def input_check(data: Dict):
    for attr in categorical_attributes:
        if data[attr] == '':
            return f"missing {attr}, please check..."

    for attr in numerical_attributes:
        try:
            dat = int(data[attr])
        except:
            return f"invalid {attr} detected, please check..."



    for t in ["Text1", "Text2", "Text3"]:
        if data[t] == '':
            return "please answer the first three quesitons..."

    text = ""
    for t in ["Text1", "Text2", "Text3", "Text4", "Text5"]:
        text = text + " " + data[t].strip()
    if len(text) < 8:
        return "please input at least 8 words..."

    return "pass"

@app.route('/')
def index():
    return render_template('show.html')

@app.route('/data',methods=['POST'])
def show():
    data = request.form.to_dict()
    data["score"] = 0
    data['level'] = "No depression detected"
    data['suggestion'] = ""

    fileData = request.files
    input_check_res = input_check(data)


    if input_check_res == "pass":
        pa_pred = float(personal_attribute_model.process_one(data)["depression"])

        text = ""
        for t in ["Text1", "Text2", "Text3", "Text4", "Text5"]:
            text = text + " " + data[t].strip()

        text_pred = float(bert_clf.process_one(text)["depression"])

        audio_pred = 0
        if fileData:

            file = request.files['fileData']
            t = time.strftime('%Y%m%d%H%M%S')
            file_path = r'static/img/' + t + file.filename
            file.save(file_path)  # save wav file
            if filetype.guess(file_path).extension == 'wav':
                audio_pred = audio_clf.process_one(file_path)
                # print(audio_pred)
            else:
                logging.info("invalid audio file! please check!")
        else:
            logging.info("no audio file.")

        data["score"] = (pa_pred + text_pred) / 2 + 0.1 * audio_pred
        data['level'] = "No depression detected"
        data['suggestion'] = "您的回答表明您的心理健康状况良好。Have a good day!"

        if pa_pred > 0.6 or text_pred > 0.5:
            data['level'] = "Mild depression detected"
            data['suggestion'] = "Your answer indicates that your mental health is not good. Please note that this type of test cannot replace" \
                                 "official medical diagnosis ." \
                                 " If you feel unwell, you should make an appointment with professional mental thrapist immediately."

        if (pa_pred > 0.6 and text_pred > 0.5) or max(pa_pred, text_pred) > 0.8:
            data['level'] = "Medium depression detected"
            data['suggestion'] = "您的回答表明您可能患有抑郁症。但请注意，此类测试不能取代实际医护人员的判断。如果您感觉不舒服，应该马上预约您的医生或心理健康专家。"

        # data['suggestion'] = ""

    else:
        data["score"] = 0
        data['level'] = "No depression detected"
        data['suggestion'] = input_check_res

    returnData = {}
    returnData['code'] = 200
    returnData['msg'] = data

    print("returnData===", returnData)
    return returnData

if __name__ == "__main__":
    #change host->aliyun delete debug=true
    app.run(host='127.0.0.1', port=5000, debug=False)

