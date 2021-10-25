# -*- coding: utf-8 -*-
import requests
import time
import json
from typing import Dict

from flask_cors import *
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
from PersonalAttributeModel import PersonalAttributeModel
app = Flask(__name__)
CORS(app, supports_credentials=True)

personal_attribute_model = PersonalAttributeModel()
personal_attribute_model.load()

numerical_attributes = [ "Age",  "education_level", "total_members","living_expenses","other_expenses"]
categorical_attributes = ["sex","Married","incoming_salary"]
def input_check(data: Dict):
    print(data)
    for attr in numerical_attributes:
        try:
            dat = int(data[attr])
        except:
            return f"invalid {attr} detected, please check..."

    for attr in categorical_attributes:
        if data[attr] == '':
            return f"invalid {attr} detected, please check..."

    return "pass"

@app.route('/')
def index():
    return render_template('show.html')

@app.route('/data',methods=['POST'])
def show():
    data = request.form.to_dict()

    res = input_check(data)
    if res == "pass":
        pred = float(personal_attribute_model.process_one(data)["depression"])
    else:
        pred = {'msg': res}
    data['img'] = ""
    fileData = request.files
    if fileData:
        fileData = request.files['fileData']
        print("fileData===",fileData)
        t = time.strftime('%Y%m%d%H%M%S')
        new_fname = r'static/img/' + t + fileData.filename
        fileData.save(new_fname)  #save img
        data['img'] = new_fname

    returnData = {}
    returnData['code'] = 200
    # returnData['msg'] = pred
    data["sex"] = pred

    returnData['msg'] = data
    print("returnData===",returnData)
    return returnData

if __name__ == "__main__":
    #change host->aliyun delete debug=true
    app.run(host='127.0.0.1', port=5000,debug=True)

