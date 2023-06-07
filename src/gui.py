"""
gui界面
"""
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from recognition import *

matplotlib.use("Qt5Agg")


def get_faces_from_image(img_path):
    """
    获取图片中的人脸
    :param img_path:
    :return:
    """
    img, img_gray, faces = face_detect(img_path, 'blazeface')
    if len(faces) == 0:
        return None
    # 遍历每一个脸
    faces_gray = []
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        face_img_gray = cv2.resize(face_img_gray, (48, 48))
        faces_gray.append(face_img_gray)
    return faces_gray


class UI(object):
    def __init__(self, form, model):
        self.model = model
        form.setObjectName("my_gui")
        form.resize(1200, 800)
        # 原图无图时显示的label
        self.label_raw_pic = QtWidgets.QLabel(form)
        self.label_raw_pic.setGeometry(QtCore.QRect(10, 30, 320, 320))
        self.label_raw_pic.setStyleSheet("background-color:#bbbbbb;")
        self.label_raw_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw_pic.setObjectName("label_raw_pic")

        # 原图下方分割线
        self.line1 = QtWidgets.QFrame(form)
        self.line1.setGeometry(QtCore.QRect(340, 30, 20, 431))
        self.line1.setFrameShape(QtWidgets.QFrame.VLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)

        # my_pic
        self.my_pic = QtWidgets.QLabel(form)
        self.my_pic.setGeometry(QtCore.QRect(10, 500, 320, 275))
        self.my_pic.setStyleSheet("background-color:#bbbbbb;")
        self.my_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.my_pic.setObjectName("my_pic")
        img = cv2.cvtColor(cv2.imread('../assets/my_pic.webp'), cv2.COLOR_BGR2RGB)
        width = int(img.shape[1] * 0.65)
        height = int(img.shape[0] * 0.65)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        self.my_pic.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                         QtGui.QImage.Format_RGB888)))
        # 结果布局设置
        self.layout_widget = QtWidgets.QWidget(form)
        self.layout_widget.setGeometry(QtCore.QRect(10, 310, 320, 240))
        self.layout_widget.setObjectName("layoutWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.layout_widget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("verticalLayout")
        # 右侧水平线
        self.line2 = QtWidgets.QFrame(self.layout_widget)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.vertical_layout.addWidget(self.line2)
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setObjectName("horizontalLayout")

        self.pushButton_select_img = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_img.setObjectName("pushButton_2")

        self.pushButton_open_camera = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_open_camera.setObjectName("pushButton_3")

        self.horizontal_layout.addWidget(self.pushButton_select_img)
        self.horizontal_layout.addWidget(self.pushButton_open_camera)

        self.vertical_layout.addLayout(self.horizontal_layout)
        self.graphicsView = QtWidgets.QGraphicsView(form)
        self.graphicsView.setGeometry(QtCore.QRect(360, 210, 800, 500))
        self.graphicsView.setObjectName("graphicsView")
        self.label_result = QtWidgets.QLabel(form)
        self.label_result.setGeometry(QtCore.QRect(361, 21, 71, 16))
        self.label_result.setObjectName("label_result")
        self.label_emotion = QtWidgets.QLabel(form)
        self.label_emotion.setGeometry(QtCore.QRect(715, 21, 71, 16))
        self.label_emotion.setObjectName("label_emotion")
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bar = QtWidgets.QLabel(form)
        self.label_bar.setGeometry(QtCore.QRect(720, 170, 100, 180))
        self.label_bar.setObjectName("label_bar")
        self.line = QtWidgets.QFrame(form)
        self.line.setGeometry(QtCore.QRect(361, 150, 800, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_rst = QtWidgets.QLabel(form)
        self.label_rst.setGeometry(QtCore.QRect(700, 50, 100, 100))
        self.label_rst.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rst.setObjectName("label_rst")

        self.pushButton_select_img.clicked.connect(self.open_file_browser)
        self.pushButton_open_camera.clicked.connect(self.open_camera)

        self.retranslate_ui(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def retranslate_ui(self, form):
        _translate = QtCore.QCoreApplication.translate
        form.setWindowTitle(_translate("my_gui", "my_gui"))
        self.label_raw_pic.setText(_translate("my_gui", "₍˄·͈༝·͈˄*₎◞ ̑̑"))
        self.pushButton_select_img.setText(_translate("my_gui", "选取图片或视频"))
        self.pushButton_open_camera.setText(_translate("my_gui", "打开摄像头"))
        self.label_result.setText(_translate("my_gui", "识别结果"))
        self.label_emotion.setText(_translate("my_gui", "null"))
        self.label_bar.setText(_translate("my_gui", "概率直方图 "))
        self.label_rst.setText(_translate("my_gui", "Result"))

    def open_camera(self):
        predict_expression_video(" ")

    # 选取图片
    def open_file_browser(self):
        # 加载模型
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片或视频",
                                                                     directory="../input/",
                                                                     filter="Images(*.jpg *.png *.mp4)")
        # 显示原图
        if file_name is not None and file_name != "":
            if str(file_name).split(".")[1] == "mp4":
                predict_expression_video(file_name)
            else:
                emotion, possibility = predict_expression(file_name, self.model)
                self.show_results(emotion, possibility)
                self.show_raw_img(file_name)

    def show_raw_img(self, filename):
        # img = cv2.imread(filename)
        img = cv2.imread("../output/rst.png")

        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 320))
        self.label_raw_pic.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))

    def show_results(self, emotion, possibility):
        # 显示表情名
        self.label_emotion.setText(QtCore.QCoreApplication.translate("my_gui", emotion))
        # 显示emoji
        if emotion != 'no':
            img = cv2.imread('../assets/icons/' + str(emotion) + '.png')
            frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (100, 100))
            self.label_rst.setPixmap(QtGui.QPixmap.fromImage(
                QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                             QtGui.QImage.Format_RGB888)))
        else:
            self.label_rst.setText(QtCore.QCoreApplication.translate("my_gui", "no result"))
        # 显示直方图
        self.show_bars(list(possibility))

    def show_bars(self, possibility):
        dr = MyFigureCanvas()
        dr.draw_(possibility)
        graphic_scene = QtWidgets.QGraphicsScene()
        graphic_scene.addWidget(dr)
        self.graphicsView.setScene(graphic_scene)
        self.graphicsView.show()


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        self.axes.bar(x, possibility, align='center')


def load_cnn_model():
    """
    载入CNN模型
    """
    model = CNN()
    model.load_weights('../models/models.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
