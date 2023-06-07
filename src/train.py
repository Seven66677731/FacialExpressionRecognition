"""
训练模块
"""
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model import CNN

his = None
# 数据集（"fer2013  jaffe  ck+"）
dataset = "fer2013"
# 训练次数
epochs = 100
# 批量大小
batch_size = 32
# 是否保存训练结果
plot_history = True


class Fer2013(object):
    def __init__(self, folder="../dataset/fer2013"):
        """
        构造函数
        """
        self.folder = folder

    def gen_train(self):
        """
        产生训练数据
        :return expressions:读取文件的顺序即标签的下标对应
        :return x_train: 训练数据集
        :return y_train： 训练标签
        """
        folder = os.path.join(self.folder, 'Training')
        # 这里原来是list出多个表情类别的文件夹，后来发现服务器linux顺序不一致，会造成问题，所以固定读取顺序
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        # 初始化训练数据集和训练标签列表为空
        x_train = []
        y_train = []
        # 遍历每个表情类别
        for i in tqdm(range(len(expressions))):
            # 忽略表情类别为contempt
            if expressions[i] == 'contempt':
                continue
            # 获取当前表情类别的文件夹路径
            expression_folder = os.path.join(folder, expressions[i])
            # 列出当前表情类别文件夹下的所有图片文件
            images = os.listdir(expression_folder)
            # 遍历当前表情类别文件夹下的每张图片
            for j in range(len(images)):
                # 加载图片并将其调整为指定大小(48x48)，转换为灰度图像
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_valid(self):
        """
        产生验证集数据
        :return:
        """
        folder = os.path.join(self.folder, 'PublicTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_valid = []
        y_valid = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_valid.append(img)
                y_valid.append(i)
        x_valid = np.array(x_valid).astype('float32') / 255.
        y_valid = np.array(y_valid).astype('int')
        return expressions, x_valid, y_valid

    def gen_test(self):
        """
        产生验证集数据
        :return:
        """
        folder = os.path.join(self.folder, 'PrivateTest')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_test = []
        y_test = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_test.append(img)
                y_test.append(i)
        x_test = np.array(x_test).astype('float32') / 255.
        y_test = np.array(y_test).astype('int')
        return expressions, x_test, y_test


class Jaffe(object):
    """
    Jaffe没有测试数据，需要自己划分
    """

    def __init__(self):
        self.folder = '../dataset/jaffe'

    def gen_train(self):
        """
        产生训练数据
        注意产生的是(213, 48, 48, 1)和(213, )的x和y，如果输入灰度图需要将x的最后一维squeeze掉
        :return:
        """
        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train


class CK(object):
    """
    CK+
    """

    def __init__(self):
        self.folder = '../dataset/ck+'

    def gen_train(self):
        """
        产生训练数据
        :return:
        """
        folder = self.folder
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                # 没有中性表情，直接跳过
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train


def plot_acc(his, ds):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['accuracy'])), his['accuracy'], label='train accuracy')
    plt.plot(np.arange(len(his['val_accuracy'])), his['val_accuracy'], label='valid accuracy')
    plt.title(ds + ' training accuracy')
    plt.legend(loc='best')
    plt.savefig('../assets/' + ds + '_his_acc.png')


def plot_loss(his, ds):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='train loss')
    plt.plot(np.arange(len(his['val_loss'])), his['val_loss'], label='valid loss')
    plt.title(ds + ' training loss')
    plt.legend(loc='best')
    plt.savefig('../assets/' + ds + '_his_loss.png')


if dataset == "fer2013":
    # 生成训练集
    expressions, x_train, y_train = Fer2013().gen_train()
    # 生成验证集
    _, x_valid, y_valid = Fer2013().gen_valid()
    # 生成测试集
    _, x_test, y_test = Fer2013().gen_test()
    # target编码
    y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)
    y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 1))))
    y_valid = np.hstack((y_valid, np.zeros((y_valid.shape[0], 1))))
    print("load fer2013 dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0],
                                                                                                 y_valid.shape[0]))
    # 定义一个输入形状为(48, 48, 1)，输出类别数为8的CNN模型
    model = CNN(input_shape=(48, 48, 1), n_classes=8)
    # 定义优化器并编译模型
    # SGD 随机梯度
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # 随机梯度下降（SGD）优化器 分类交叉熵损失函数 模型评估指标（准确率）
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # 定义回调函数用于模型训练
    callback = [
        # 训练结果
        # monitor = 'val_acc'：监控的指标为验证集的准确率。
        # verbose = True：显示关于保存模型权重的信息。
        # save_best_only = True：只保存在验证集上性能最好的模型权重。
        # save_weights_only = True：仅保存模型的权重而不保存整个模型。
        ModelCheckpoint('../models/models.h5', monitor='val_acc', verbose=True, save_best_only=False,
                        save_weights_only=True)]
    # 增强操作
    # rotation_range = 10：随机旋转图像的角度范围为-10到+10度之间。
    # width_shift_range = 0.05：随机水平平移图像的宽度范围为图像宽度的5%。
    # height_shift_range = 0.05：随机垂直平移图像的高度范围为图像高度的5%。
    # horizontal_flip = True：随机以50 % 的概率水平翻转图像。
    # shear_range = 0.2：随机错切变换图像的错切强度范围为0.2。
    # zoom_range = 0.2：随机缩放图像的尺寸范围为图像尺寸的20 %。

    train_generator = ImageDataGenerator(rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=batch_size)
    # 训练模型
    history_fer2013 = model.fit_generator(train_generator,
                                          steps_per_epoch=len(y_train) // batch_size,
                                          epochs=epochs,
                                          validation_data=valid_generator,
                                          validation_steps=len(y_valid) // batch_size,
                                          callbacks=callback)
    his = history_fer2013

    # 测试模型
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print("test accuracy", np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])

elif dataset == "jaffe":
    expressions, x, y = Jaffe().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y = np.hstack((y, np.zeros((y.shape[0], 1))))

    # 划分训练集验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2023)
    print("load jaffe dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0],
                                                                                               y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=5,
                                         width_shift_range=0.01,
                                         height_shift_range=0.01,
                                         horizontal_flip=True,
                                         shear_range=0.1,
                                         zoom_range=0.1).flow(x_train, y_train, batch_size=batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=batch_size)

    model = CNN()

    sgd = Adam(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [
        # EarlyStopping(monitor='val_loss', patience=50, verbose=True),
        # ReduceLROnPlateau(monitor='lr', factor=0.1, patience=15, verbose=True),
        ModelCheckpoint('../models/models.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)]
    history_jaffe = model.fit(train_generator, steps_per_epoch=len(y_train) // batch_size, epochs=epochs,
                              validation_data=valid_generator, validation_steps=len(y_valid) // batch_size,
                              callbacks=callback)
    his = history_jaffe
else:
    expr, x, y = CK().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 划分训练集验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    print("load CK+ dataset successfully, it has {} train images and {} valid images".format(y_train.shape[0],
                                                                                             y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=batch_size)
    model = CNN()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [
        ModelCheckpoint('../models/models.h5', monitor='val_acc', verbose=True, save_best_only=False,
                        save_weights_only=True)]
    history_ck = model.fit_generator(train_generator, steps_per_epoch=len(y_train) // batch_size, epochs=epochs,
                                     validation_data=valid_generator, validation_steps=len(y_valid) // batch_size,
                                     callbacks=callback)
    his = history_ck

if plot_history:
    plot_loss(his.history, dataset)
    plot_acc(his.history, dataset)
