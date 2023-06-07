"""
数据预处理
"""
# encoding:utf-8
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'normal',  # 中性
}


# 创建文件夹
def createDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# fer2013数据集预处理
def saveImageFromFer2013(file):
    imageCount = 0
    cvss = os.listdir(file)
    for cvs in cvss:
        # 读取csv文件
        cvs = os.path.join(file, cvs)
        faces_data = pd.read_csv(cvs)
        imageCount = 0
        # 遍历csv文件内容，并将图片数据按分类保存
        for index in tqdm(range(len(faces_data))):
            # 解析每一行csv文件内容
            emotion_data = faces_data.loc[index][0]
            image_data = faces_data.loc[index][1]
            # 将图片数据转换成48*48
            data_array = list(map(float, image_data.split()))
            data_array = np.asarray(data_array)
            image = data_array.reshape(48, 48)

            # 选择分类，并创建文件名
            dirName = "../data/fer2013/" + str(cvs).split("\\")[1].split(".")[0]
            emotionName = emotions[str(emotion_data)]

            # 图片要保存的文件夹
            imagePath = os.path.join(dirName, emotionName)

            # 创建“用途文件夹”和“表情”文件夹
            createDir(dirName)
            createDir(imagePath)

            # 图片文件名
            imageName = os.path.join(imagePath, str(index) + '.jpg')

            cv2.imwrite(imageName, image)
            imageCount = index
    print('总共有' + str(imageCount) + '张图片')


def countImage(filePath):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    listdir = os.listdir(filePath)
    for i in listdir:
        if os.path.isdir(os.path.join(filePath, i)):
            path_join = os.path.join(filePath, i)
            x = os.listdir(path_join)
            y = []
            for emotion in x:
                path = os.path.join(path_join, emotion)
                num = len(os.listdir(path))
                y.append(num)
            for i in range(len(x)):
                plt.bar(x[i], y[i])
            plt.title(str(path_join).split("\\")[1] + "数据分布")
            plt.show()


if __name__ == '__main__':
    saveImageFromFer2013('../data/fer2013')
    countImage("../data/fer2013")
