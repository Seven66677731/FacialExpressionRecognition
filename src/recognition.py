"""
表情识别模块
"""
import os
import cv2
import numpy as np
from blazeface import blaze_detect
from model import CNN


def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN()
    model.load_weights('../models/models.h5')
    return model


def face_detect(img_path, model_selection="default"):
    """
    检测测试图片的人脸
    """
    # 读取图像
    img = cv2.imread(img_path)
    # 将图像转换为灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用Haar级联分类器模型进行人脸检测
    if model_selection == "default":
        face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30)
        )
    # 使用BlazeFace模型进行人脸检测
    elif model_selection == "blazeface":
        faces = blaze_detect(img)
    else:
        raise NotImplementedError("this face detector is not supported now!!!")

    return img, img_gray, faces


def index2emotion(index=0, kind='cn'):
    """
    根据表情下标返回表情字符串
    """
    emotions = {
        '发怒': 'anger',
        '厌恶': 'disgust',
        '恐惧': 'fear',
        '开心': 'happy',
        '伤心': 'sad',
        '惊讶': 'surprised',
        '中性': 'neutral',
        '蔑视': 'contempt'

    }
    if kind == 'cn':
        return list(emotions.keys())[index]
    else:
        return list(emotions.values())[index]


def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    """
    :param img:
    :param text:
    :param left:
    :param top:
    :param text_color:
    :param text_size
    :return:
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        "./assets/simsun.ttc", text_size, encoding="utf-8")  # 使用宋体
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """
    # 归一化，将像素值缩放到[0, 1]范围
    face_img = face_img / 255.
    # 调整图像大小
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    # 添加原始图像和多个增广图像到列表中
    resized_images.append(face_img[:, :])  # 原始图像 0
    resized_images.append(face_img[2:45, :])  # 剪裁上下边缘的图像 1
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))  # 水平翻转的图像 2

    # 调整图像大小并添加通道维度
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression(img_path, model):
    """
    对图中n个人脸进行表情预测
    :param img_path: 图片的完整路径
    :param model: 表情预测模型
    :return: 预测的主要表情和各个表情的概率
    """
    border_color = (0, 255, 0)  # 绿框框
    font_color = (255, 255, 255)  # 白字字

    # 进行人脸检测，使用BlazeFace模型
    img, img_gray, faces = face_detect(img_path, 'blazeface')
    # 没有检测到人脸
    if len(faces) == 0:
        return 'no', [0, 0, 0, 0, 0, 0, 0, 0]

    emotions = []
    result_possibilities = []
    # 遍历每一个脸
    for (x, y, w, h) in faces:
        # 获取人脸区域的灰度图像
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        # 生成人脸图像
        faces_img_gray = generate_faces(face_img_gray)
        # 预测表情
        results = model.predict(faces_img_gray)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index, 'en')

        # 绘制人脸框和表情文字
        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
        img = cv2_img_add_text(img, emotion, x + 30, y + 30, font_color, 20)

        emotions.append(emotion)
        result_possibilities.append(result_sum)
    # 保存结果图像
    if not os.path.exists("../output"):
        os.makedirs("../output")
    cv2.imwrite('../output/rst.png', img)
    return emotions[0], result_possibilities[0]


def predict_expression_video(filename):
    """
    实时预测
    :return:
    """
    # 参数设置
    model = load_model()

    border_color = (0, 255, 0)  # 绿框框
    font_color = (255, 255, 255)  # 白字字
    capture = cv2.VideoCapture(0)  # 指定0号摄像头

    if not str(filename) == " ":
        capture = cv2.VideoCapture(filename)
    else:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
        retval, frame = capture.read()
        if not retval:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 灰度化
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = blaze_detect(frame)
        # 如果检测到人脸
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                # 从灰度图像中提取脸部区域
                face = frame_gray[y: y + h, x: x + w]
                # 将探测到的人脸进行增广
                faces = generate_faces(face)
                # 使用模型对脸部数据进行预测
                results = model.predict(faces)
                # 对结果进行求和并重新调整数组形状
                result_sum = np.sum(results, axis=0).reshape(-1)
                # 获取具有最高值的索引
                label_index = np.argmax(result_sum, axis=0)
                # 将索引转换为情绪标签
                emotion = index2emotion(label_index)
                # 在原始图像上绘制脸部的矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, font_color, 20)
        cv2.imshow("press esc to exit", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)  # 等待30ms，返回ASCII码

        # 如果输入esc则退出循环
        if key == 27:
            break
    capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 销毁窗口
