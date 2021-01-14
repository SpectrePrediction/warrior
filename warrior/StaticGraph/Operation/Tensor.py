import numpy as np


class Tensor(object):
    """
    节点中存放的Tensor流，表示节点之间的关系或者数据的流动关系
    举例：1 + 1 = 2 （假设命名x + y = z）

    """
    def __init__(self, *input_nodes, dtype, name):
        """
        Tensor流，表示节点之间的流动关系，存在于节点中
        :param input_nodes: 上一个或多个节点,通过*可传入多个
        :type input_nodes: 存放节点类型的元祖
        :param dtype: 此节点的类型（由节点传入）
        :param name: 此节点的名字（由节点传入）
        """
        # Tensor流的上一个或多个节点
        self.input_nodes = input_nodes

        # 用来存放以此Tensor流为输入的节点（下一个节点）
        self.output_nodes = []

        # Tensor中存放的输出的值
        # （在run之前，这个值是空的）
        self.output_value = None

        # Tensor命名(节点名）
        self.name = str(name)

        # 节点类型（不同类型不允许相加）
        self.dtype = str(dtype)

        self.shape = None
        self.shape = self.get_shape()

    def __str__(self):
        return "<warrior.Tensor name:'" + self.name + \
               "' value: " + str(self.output_value) + \
               " dtype: " + str(self.dtype) + \
               " shape: " + str(self.get_shape()) + ">"

    def get_save_information(self):
        """
        优化后的保存信息
        可以大幅减少保存模型的大小
        这与str(self)区分开来
        仅仅保存需要用到的name和type
        :return: str
        """
        return "name:'" + self.name + \
               "' dtype: " + str(self.dtype)

    def get_shape(self):
        """
        取得Tensor中节点值的维度
        :return: 相应维度 tuple（标量为空元祖）
        """
        shape = None
        if self.shape != shape:
            shape = self.shape
        if self.output_value is not None:
            shape = np.shape(self.output_value)
        return shape


if __name__ == '__main__':
    testT = Tensor([1], dtype=int, name="test")
    testT.output_value = 10
    print(testT)
