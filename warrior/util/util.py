import numpy as np
import os
import struct


def one_hot(labels, label_class):
    """
    将列表转换成一热编码
    :param labels:  list
    :param label_class: 种类数量
    :return:np.ndarray
    """
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(label_class)] for j in range(len(labels))])
    return one_hot_label


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


class MnistReader:

    def __init__(self, mnist_path, using_onehot=True, kind='train'):
        self.mnist_img, self.mnist_label = load_mnist(mnist_path, kind)
        self.img_list, self.label_list = self.__pretreatment_and_return_list(self.mnist_img, self.mnist_label)
        if using_onehot:
            self.label_list = one_hot(self.label_list, 10)
        self.total = len(self.img_list)
        self.number = 0

    def __pretreatment_and_return_list(self, mnist_img, mnist_label):
        img_list = []
        label_list = []

        for i in range(mnist_img.shape[0]):
            input_x = mnist_img[i]
            input_y = mnist_label[i]
            input_x = np.reshape(input_x, (28, 28))
            img_list.append(input_x)
            label_list.append(input_y)
        return img_list, label_list

    def next_train_data(self, batch_size=1, *, number=None):
        """
        获取下一批训练集数据
        :param number:如果填入，则返回相应批次的训练集
        注：如果大于批次，将返回第0批
        :return:处理好的训练集和一热编码的标签
        """
        all_epoch = self.total // batch_size
        if number:
            _number = number
        else:
            _number = self.number
            self.number += 1

        if _number >= all_epoch:
            _number = self.number = 0

        batch_x = self.img_list[_number * batch_size: (_number + 1) * batch_size]
        batch_y = self.label_list[_number * batch_size: (_number + 1) * batch_size]

        return batch_x, batch_y


def argmax(array_list, axis=None, out=None):
    return np.argmax(array_list, axis, out)
