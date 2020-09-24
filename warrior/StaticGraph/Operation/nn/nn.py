from ..Operation import *


class Sigmoid(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.tensor.output_value = 1/(1 + np.exp(-x.tensor.output_value))
            # if x.tensor.output_value >= 0:
            #     self.tensor.output_value = 1 / (1 + np.exp(-x.tensor.output_value))
            # else:
            #     self.tensor.output_value = np.exp(-x.tensor.output_value) / (1 + np.exp(-x.tensor.output_value))
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        if x.tensor.output_value >= 0:
            self.tensor.output_value = 1/(1 + np.exp(-x.tensor.output_value))
        else:
            self.tensor.output_value = np.exp(-x.tensor.output_value) / (1 + np.exp(-x.tensor.output_value))
        return self.tensor.output_value

    def compute_gradient(self, grad=None):

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)
        return grad*self.tensor.output_value*(1 - self.tensor.output_value)


"""
损失函数
"""


def get_loss_form_str(optimizer, *, logits, labels):
    return globals()[optimizer](logits, labels)


class SSELoss(object):
    """
    和方差损失
    """
    def __new__(cls, logits, labels):
        """
        构造之前，使用魔术方法
        返回和方差的节点，而不是类
        :param logits: 预测值
        :param labels: 标签值
        :type logits labels: Operation
        :return: Operation
        """
        try:
            assert isinstance(logits, Operation)
            assert isinstance(labels, Operation)
        except AssertionError as err:
            Error("损失函数应当传入节点. in {}".format(cls), is_raise=True, exc_info=err)

        cls.logits = logits
        cls.labels = labels

        return cls.forward(cls)

    def forward(self):
        return ReduceSum(Square(self.logits - self.labels))





