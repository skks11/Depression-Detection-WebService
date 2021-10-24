from numpy import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os
import math
from natsort import natsorted
import cv2
import dlib
import PIL
from PIL import Image

import warnings
import pickle
from sklearn.svm import SVC

warnings.simplefilter('ignore', np.RankWarning) # 关闭警告：RankWarning: Polyfit may be poorly conditioned

class singleModel(object):
    def __init__(self):
        self.model1 = None
        # put your model configs here
        # self.train = config["train_file"]
        # ....
        self.dlib = dlib
        self.predictor_path = r"/content/drive/MyDrive/data/raw_data/shape_predictor_68_face_landmarks.dat"
        self.predictor = self.dlib.shape_predictor(self.predictor_path)
        self.detector = self.dlib.get_frontal_face_detector() # 初始化dlib人脸检测器
        


    def pred(self, model1, X_t2, image_path):
        """
        return predictions: 0-1
        """
        X_t2 = self.feature_extract(image_path)
        #predicted = model1.predict(X_t2)
        predicted = model1.predict_proba(X_t2)  # 输出分类概率
        return predicted

    def feature_extract(self,image_path):
        """
        extract feature from raw data
        """
        X_t2 = self.getImage(image_path)  # 读取视频目录下的所有视频的人脸特征
        #print("X_t2:", X_t2)
        return X_t2

    def train(X, Y, X_t2, Y_t2):
        # sv.fit(X, Y)
        # sv.score(X_t2, Y_t2)
        pass

    def save():
        """
        save model
        """
        # s = pickle.dumps(sv)
        # f = open('/content/drive/MyDrive/svm.model', "wb+")
        # f.write(s)
        # f.close()
        # print("Done\n")
        pass

    def load(self):
        """
        load model
        """
        f2 = open('/content/drive/MyDrive/svm.model', 'rb')
        s2 = f2.read()
        model1 = pickle.loads(s2)
        return model1

    def convert2onnx():
        """
        if inference time(except video model) is greater than 10ms on your machine:
            1. reduce model size
            2. use onnx https://github.com/onnx/tutorials
        """
        pass

    def process_one(self,image_path,model1):
        """
        input: raw data
        output: prediction
        """
        temp = dict(depression=0.0, nondepression=0.0)

        #image_path = '/content/drive/MyDrive/data/raw_data/all_video/test1.jpg'
        X_t2 = self.feature_extract(image_path)
        #model1 = self.load()
        result = self.pred(model1, X_t2, image_path)

        result = np.array(result)

        temp = dict(depression=result[0][0], nondepression=result[0][1])
        
        print(temp)


    #提取特征
    def get_landmarks(self,image_path):
        """先将图片裁剪，仅保留人脸部分，再提取出坐标特征

        :param image: 图片
        :return: 68个人脸坐标特征
        """
        predictor_path = r"/content/drive/MyDrive/data/raw_data/shape_predictor_68_face_landmarks.dat"
        self.predictor = self.dlib.shape_predictor(predictor_path)
        self.detector = self.dlib.get_frontal_face_detector() # 初始化dlib人脸检测器

      
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("gray:", gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        detections = self.detector(clahe_image, 1)

        # print("detection:", detections)
        for k, d in enumerate(detections):
            shape = self.predictor(clahe_image, d)
            xlist = []
            ylist = []
            landmarks = []
            for i in range(0, 68):
                cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)

                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            x_max = np.max(xlist)
            x_min = np.min(xlist)
            y_max = np.max(ylist)
            y_min = np.min(ylist)
            cv2.rectangle(clahe_image, (int(x_min), int(y_min - ((ymean - y_min) / 3))), (int(x_max), int(y_max)),
                          (255, 150, 0), 2)

            cv2.circle(clahe_image, (int(xmean), int(ymean)), 1, (0, 255, 255), thickness=2)

            x_start = int(x_min)
            y_start = int(y_min - ((ymean - y_min) / 3))
            w = int(x_max) - x_start
            h = int(y_max) - y_start

            crop_img = image[y_start:y_start + h, x_start:x_start + w]

            # print("crop_img:", crop_img)

        if len(detections) > 0:
            mywidth = 255
            hsize = 255
            cv2.imwrite('/content/drive/MyDrive/data/raw_data/all_video/crop_img.png', crop_img)
            img = Image.open('/content/drive/MyDrive/data/raw_data/all_video/crop_img.png')
            img = img.resize((mywidth, hsize), PIL.Image.ANTIALIAS)
            img.save('/content/drive/MyDrive/data/raw_data/all_video/resized.png')

            image_resized = cv2.imread('/content/drive/MyDrive/data/raw_data/all_video/resized.png')

            # print("image_resized:", image_resized)
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray)
            detections = self.detector(clahe_image, 1)

            # print("detections:", detections)

            for k, d in enumerate(detections):
                shape = self.predictor(clahe_image, d)
                xlist = []
                ylist = []
                for i in range(0, 68):
                    cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)

                    xlist.append(float(shape.part(i).x))
                    ylist.append(float(shape.part(i).y))

                xmean = np.mean(xlist)
                ymean = np.mean(ylist)
                x_max = np.max(xlist)
                x_min = np.min(xlist)
                y_max = np.max(ylist)
                y_min = np.min(ylist)
                cv2.rectangle(clahe_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 150, 0), 2)

                cv2.circle(clahe_image, (int(xmean), int(ymean)), 1, (0, 255, 255), thickness=2)

                xlist = np.array(xlist, dtype=np.float64)
                ylist = np.array(ylist, dtype=np.float64)

            if len(detections) > 0:
                return xlist, ylist
            else:
                xlist = np.array([])
                ylist = np.array([])
                return xlist, ylist
        else:
            xlist = np.array([])
            ylist = np.array([])
            return xlist, ylist

    def extract_AU(self, xlist, ylist):
        """

        :param xlist: x坐标值
        :param ylist: y坐标值
        :return: 人脸情感特征AUs
        """
        AU_feature = []

        AU1_r_x = xlist[22:25]
        AU1_r_y = ylist[22:25]
        AU1_r_x, AU1_r_y = self.linear_interpolation(AU1_r_x, AU1_r_y)

        #print("AU1_r_x:",self.get_average_curvature(AU1_r_x, AU1_r_y))
        AU_feature = AU_feature + [self.get_average_curvature(AU1_r_x, AU1_r_y)]

        #print("AU_feature:", AU_feature)

        AU2_r_x = xlist[24:27]
        AU2_r_y = ylist[24:27]
        AU2_r_x, AU2_r_y = self.linear_interpolation(AU2_r_x, AU2_r_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU2_r_x, AU2_r_y)]

        AU5_r_x = xlist[42:46]
        AU5_r_y = ylist[42:46]
        AU5_r_x, AU5_r_y = self.linear_interpolation(AU5_r_x, AU5_r_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU5_r_x, AU5_r_y)]

        AU9_x = xlist[31:36]
        AU9_y = ylist[31:36]
        AU9_x, AU9_y = self.linear_interpolation(AU9_x, AU9_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU9_x, AU9_y)]

        AU10_x = np.append(xlist[48:51], xlist[52:55])
        AU10_y = np.append(ylist[48:51], ylist[52:55])
        AU10_x, AU10_y = self.linear_interpolation(AU10_x, AU10_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU10_x, AU10_y)]

        AU12_r_x = [xlist[54]] + [xlist[64]] + [xlist[65]]
        AU12_r_y = [ylist[54]] + [ylist[64]] + [ylist[65]]
        AU12_r_x, AU12_r_y = self.linear_interpolation(AU12_r_x, AU12_r_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU12_r_x, AU12_r_y)]

        AU20_x = xlist[55:60]
        AU20_y = ylist[55:60]
        AU20_x, AU20_y = self.linear_interpolation(AU20_x, AU20_y)
        AU_feature = AU_feature + [self.get_average_curvature(AU20_x, AU20_y)]

        Norm_AU_feature = (AU_feature - np.min(AU_feature)) / np.ptp(AU_feature)

        #print("Norm_AU_feature:", Norm_AU_feature)

        return Norm_AU_feature

    def linear_interpolation(self, xlist, ylist):
        xlist = np.array(xlist, dtype=np.float64)
        ylist = np.array(ylist, dtype=np.float64)
        x_new = np.array([])
        y_new = np.array([])
        for i in range(len(xlist) - 1):
            x_new = np.concatenate((x_new, [(xlist[i] + xlist[i + 1]) / 2.0]))
            y_new = np.concatenate((y_new, [(ylist[i] + ylist[i + 1]) / 2.0]))
        xlist = np.append(xlist, x_new)
        ylist = np.append(ylist, y_new)
        return xlist, ylist

    def get_average_curvature(self, AU_xlist, AU_ylist):
        K = []
        Z = np.polyfit(AU_xlist, AU_ylist, 4)
        P = np.poly1d(Z)
        P_1 = np.poly1d.deriv(P)
        P_2 = np.poly1d.deriv(P_1)
        for i in range(len(AU_xlist)):
            Y = 1 + math.pow(P_1(AU_xlist[i]), 2)
            Y = math.pow(Y, 1.5)
            K.append(P_2(AU_xlist[i]) / Y)
        m_K = np.mean(K)
        return m_K

    def getImage(self,image_path):
        X_t2 = []
        [xlist, ylist] = self.get_landmarks(image_path)

        X_t2 = self.extract_AU(xlist, ylist)

        # for root, dirs, files in os.walk(image_path):
        #     files = natsorted(files)  # 排好序
        #     for i in range(len(files)):
        #         # 初始化窗口
        #         image_path = image_path + files[i]

        #         print("image_path:", image_path)
        #         [xlist, ylist] = get_landmarks(image_path)
        #         m = extract_AU(xlist,ylist)
        #         X_t2 = X_t2.append(m)

        X_t2 = np.asarray(X_t2, dtype=np.float32)
        X_t2 = X_t2.reshape(1,-1)
        #print("AU_feature:", X_t2)
        return X_t2

#model = singleModel('/content/drive/MyDrive/data/raw_data/all_video/test1.jpg',r"/content/drive/MyDrive/data/raw_data/shape_predictor_68_face_landmarks.dat")
model = singleModel()
model1 = model.load()
model.process_one('/content/drive/MyDrive/data/raw_data/all_video/test1.jpg',model1)
