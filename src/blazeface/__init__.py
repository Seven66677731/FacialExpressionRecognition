# 人脸识别相关代码
# 代码来自 https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference
import numpy as np
from .blazeFaceDetector import blazeFaceDetector


def xyxy_to_xywh(xyxy):
    """
    将边界框坐标从 [x1, y1, x2, y2] 格式转换为 [x1, y1, w, h] 格式
    :param xyxy: 边界框坐标
    :return: 转换后的边界框坐标
    """
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def blaze_detect(image_rgb, scoreThreshold=0.7, iouThreshold=0.3, modelType="back"):
    """
    使用BlazeFace模型检测图像中的人脸
    :param image_rgb: RGB格式的输入图像
    :param scoreThreshold: 目标置信度阈值，默认为0.7
    :param iouThreshold: IOU（交并比）阈值，默认为0.3
    :param modelType: 模型类型，默认为"back"
    :return: 检测到的人脸边界框信息
    """
    # 获取图像的高度、宽度和通道数
    h, w, c = image_rgb.shape
    # 初始化BlazeFace人脸检测器
    faceDetector = blazeFaceDetector(modelType, scoreThreshold, iouThreshold)
    # 检测人脸
    results = faceDetector.detectFaces(image_rgb)
    # 获取检测结果中的边界框、关键点和置信度
    boundingBoxes = results.boxes
    keypoints = results.keypoints
    scores = results.scores
    bboxes = []

    # 添加边界框和关键点信息
    for boundingBox, keypoints, score in zip(boundingBoxes, keypoints, scores):
        x1 = (w * boundingBox[0]).astype(int)
        x2 = (w * boundingBox[2]).astype(int)
        y1 = (h * boundingBox[1]).astype(int)
        y2 = (h * boundingBox[3]).astype(int)
        bboxes.append([x1, y1, x2, y2])
    if len(bboxes) > 0:
        bboxes = np.array(bboxes).astype('int')
        bboxes = xyxy_to_xywh(np.array(bboxes))
        bboxes[:, 2] = bboxes[:, 2] * 1.1
        bboxes[:, 3] = bboxes[:, 3] * 1.1
    else:
        bboxes = None
    # 返回检测到的人脸边界框信息
    return bboxes
