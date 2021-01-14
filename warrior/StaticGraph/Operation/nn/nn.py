from ..Operation import *


class Conv2d(Operation):

    def __init__(self, x, filter_num, kernel_size, padding=0, stride=1, bias=True, data_format='NCHW', name=None):
        """
        [n, c, h, w]
        :param x: 应当是一个值类型为numpy.array的节点
        :param filter_num:
        :param kernel_size:
        :param padding:
        :param stride:
        :param bias:
        :param data_format:'NCHW' 'NHWC' 可以打乱顺序大小写等，通用 支持3位'CHW'
        :param name:
        """
        super(self.__class__, self).__init__(x, name=name)
        self.filter_num = filter_num

        data_format = data_format.upper()
        try:
            assert data_format.__len__() == 3 or data_format.__len__() == 4, "data_format应当是3位或者4位"
        except AssertionError:
            Error("data_format : {} 应当是一个4位字母或者3位字母的字符串"
                  "节点 {} \ndata_format得到 {}".format(data_format, str(self.tensor), data_format.__len__()),
                  is_raise=True, exc_info=AssertionError)

        self.format_num = data_format.__len__()

        self.transpose_index = (
            data_format.index("N"), data_format.index("C"),
            data_format.index("H"), data_format.index("W")
        ) if self.format_num == 4 else (
            data_format.index("C"), data_format.index("H"), data_format.index("W")
        )

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        padding = (padding, padding) if isinstance(padding, int) else padding
        self.padding = (
            (0, 0), (0, 0),
            padding, padding
        ) if self.format_num == 4 else (
            (0, 0), padding, padding
        )

        self.stride = (stride, stride) if isinstance(stride, int) else stride

        self._param = dict()
        # 修改权重，以应对没预热无法加载权重
        self._param["need_init"] = True
        self._param["weight"] = Variable(np.zeros(1), name=self.tensor.name.split("in")[0][:-1] + "_weight")
        self._param["weight"].tensor.output_nodes.append(self)

        # 选择了没有channel参数的版本，那就在有数据的时候初始化
        # self._param["weight"] = Variable(np.random.normal(loc=0.0, scale=1.0, size=(
        #     input_channel*self.kernel_size[0]*self.kernel_size[1], filter_num)
        # ), name=None)
        if bias:
            self._param["bias"] = Variable(np.random.normal(loc=0.0, scale=0.001, size=(1, filter_num)),
                                           name=self.tensor.name.split("in")[0][:-1] + "_bias")
            self._param["bias"].tensor.output_nodes.append(self)
        else:
            self._param["bias"] = None

        self.tensor.input_nodes = (self.tensor.input_nodes[0], self._param["weight"], self._param["bias"]) if \
            self._param["bias"] else (self.tensor.input_nodes[0], self._param["weight"])

        # 减少内存多次开销
        self.out = None
        # self.y_matrix_array = None
        self.grad = None
        self.x_matrix = None
        self.inputs_x_shape = None
        self.out_h = None
        self.out_w = None
        self.out_array = None
        self.out_count = None

    def compute_output(self):
        try:
            x = self.tensor.input_nodes[0]

            inputs_x = x.tensor.output_value
            inputs_x = np.transpose(inputs_x, self.transpose_index)
            self.inputs_x_shape = inputs_x.shape
            padding_x = np.pad(inputs_x, self.padding)

            batch_size, kernel_c, image_h, image_w = padding_x.shape if padding_x.shape.__len__() == 4 else \
                (1, *padding_x.shape)

            if self._param["need_init"]:
                self._param["weight"].tensor.output_value = np.random.normal(loc=0.0, scale=0.001, size=(
                    kernel_c*self.kernel_size[0]*self.kernel_size[1], self.filter_num))

                self._param["weight"].auto_output()
                if self._param["bias"] is not None:
                    self._param["bias"].auto_output()

            self.out_h = (image_h - self.kernel_size[0]) // self.stride[0] + 1
            self.out_w = (image_w - self.kernel_size[1]) // self.stride[1] + 1

            self.out = np.zeros((self.out_h * self.out_w, self.kernel_size[0] * self.kernel_size[1] * kernel_c)) if \
                self.out is None else self.out

            y_matrix_array = np.zeros((batch_size, self.filter_num, self.out_h, self.out_w))

            self.x_matrix = []
            for batch_index in range(batch_size):
                batch_x = padding_x[batch_index]
                for idx_h, i in enumerate(range(0, image_h - self.kernel_size[0] + 1, self.stride[0])):
                    for idx_w, j in enumerate(range(0, image_w - self.kernel_size[1] + 1, self.stride[1])):

                        self.out[idx_h * self.out_w + idx_w, :] = \
                            batch_x[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]].reshape(1, -1)

                self.x_matrix.append(self.out.copy())
                y_matrix = np.dot(self.out, self._param["weight"].tensor.output_value)
                if self._param["bias"]:
                    y_matrix += self._param["bias"].tensor.output_value

                for c in range(y_matrix.shape[1]):
                    y_matrix_array[batch_index, c] = y_matrix[:, c].reshape(self.out_h, self.out_w)

            self.x_matrix = np.asarray(self.x_matrix)
            self.tensor.output_value = y_matrix_array

        except ValueError as err:
            print(err)
            Error("节点 {} \n错误信息 {} \n请确保数据输入格式与data_format格式一致".format(str(self.tensor), str(err)),
                  is_raise=True, exc_info=ValueError)

        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)

        return self.tensor.output_value

    def auto_output(self):
        try:
            x = self.tensor.input_nodes[0]
            if x.tensor.output_value is None:
                x.auto_output()

            self.compute_output()

            # inputs_x = x.tensor.output_value
            # inputs_x = np.transpose(inputs_x, self.transpose_index)
            # padding_x = np.pad(inputs_x, self.padding)
            #
            # batch_size, kernel_c, image_h, image_w = padding_x.shape if padding_x.shape.__len__() == 4 else \
            #     (1, *padding_x.shape)
            #
            # if self._param["weight"] is None:
            #     self._param["weight"] = Variable(np.random.normal(loc=0.0, scale=0.001, size=(
            #         kernel_c*self.kernel_size[0]*self.kernel_size[1], self.filter_num)
            #     ), name=self.tensor.name.split("in")[0] + "weight")
            #     # 测试使用
            #     # self._param["weight"] = Variable(np.ones((
            #     #     kernel_c * self.kernel_size[0] * self.kernel_size[1], self.filter_num)
            #     # ), name=self.tensor.name + "weight")
            #     self._param["weight"].auto_output()
            #     if self._param["bias"] is not None:
            #         self._param["bias"].auto_output()
            #
            # out_h = (image_h - self.kernel_size[0]) // self.stride[0] + 1
            # out_w = (image_w - self.kernel_size[1]) // self.stride[1] + 1
            # # out = np.zeros((out_h*out_w, kernel_h*kernel_w*kernel_c))
            # out = np.empty((out_h * out_w, self.kernel_size[0] * self.kernel_size[1] * kernel_c))
            # y_matrix_array = np.empty((batch_size, self.filter_num, out_h, out_w))
            #
            # # out_array = np.empty((kernel_c, image_h, image_w))
            # # y = []
            # for batch_index in range(batch_size):
            #     batch_x = padding_x[batch_index]
            #     for idx_h, i in enumerate(range(0, image_h - self.kernel_size[0] + 1, self.stride[0])):
            #         for idx_w, j in enumerate(range(0, image_w - self.kernel_size[1] + 1, self.stride[1])):
            #
            #             out[idx_h * out_w + idx_w, :] = \
            #                 batch_x[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]].reshape(1, -1)
            #
            #     y_matrix = np.dot(out, self._param["weight"].tensor.output_value)
            #     if self.bias:
            #         y_matrix += self._param["bias"].tensor.output_value
            #
            #     # 将out变回原图像
            #     # for i in range(out_h):
            #     #     for j in range(out_w):
            #     #         out_array[:, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
            #     #         j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]] = \
            #     #             out[i * out_w + j].reshape(-1, self.kernel_size[0], self.kernel_size[1])
            #
            #     # 测试中在通道为1的情况下还有问题,感觉是个简单的问题，太晚了，搁置先，启用另一种
            #     # 简单错误，已修复
            #     for c in range(y_matrix.shape[1]):
            #         y_matrix_array[batch_index, c] = y_matrix[:, c].reshape(out_h, out_w)
            #
            # # 速度上差距不大，保留写法, 稍微不好维护
            # #     y_i = []
            # #     for c in range(y_matrix.shape[1]):
            # #         y_i.append(y_matrix[:, c].reshape(out_h, out_w))
            # #     y.append(np.asarray(y_i))
            # # y = np.asarray(np.asarray(y))
            #
            # self.tensor.output_value = y_matrix_array

        except ValueError as err:
            print(err)
            Error("节点 {} \n错误信息 {} \n请确保数据输入格式与data_format格式一致".format(str(self.tensor), str(err)),
                  is_raise=True, exc_info=ValueError)

        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        d_weight = np.zeros_like(self._param["weight"].tensor.output_value)
        d_bias = np.zeros_like(self._param["bias"].tensor.output_value) if self._param["bias"] else None
        d_x = np.zeros_like(self.out)

        grad_b, grad_c, grad_h, grad_w = grad.shape
        grad_matrix = np.zeros((grad_b, grad_h*grad_w, grad_c))
        for i in range(grad_b):
            for j in range(grad_c):
                grad_matrix[i, :, j] = grad[i, j, :, :].reshape((-1))

        for idx, x_matrix in enumerate(self.x_matrix):
            d_weight += np.dot(x_matrix.T, grad_matrix[idx, :, :])
            d_x += np.dot(grad_matrix[idx, :, :], self._param["weight"].tensor.output_value.T)
            if self._param["bias"]:
                d_bias += np.mean(grad_matrix[idx, :, :], axis=0, keepdims=True)

        d_weight /= (grad_b * self.out_h * self.out_w)  # *self.__kernel_size[1]**self.__kernel_size[2])
        d_x /= grad_b
        if self._param["bias"]:
            d_bias /= grad_b

        shape = self.inputs_x_shape if self.inputs_x_shape.__len__() == 4 else (1, *self.inputs_x_shape)
        self.out_array = np.zeros(shape) if self.out_array is None else self.out_array
        self.out_count = np.zeros(shape) if self.out_count is None else self.out_count

        for batch in range(shape[0]):
            for i in range((shape[2]-self.kernel_size[0]) // self.stride[0] + 1):
                for j in range((shape[3]-self.kernel_size[1]) // self.stride[1] + 1):

                    self.out_count[batch, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                              j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]] += 1

                    self.out_array[batch, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                              j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]] +=\
                        d_x[i * self.out_w + j].reshape(-1, self.kernel_size[0], self.kernel_size[1])

        self.out_count[self.out_count == 0] = 1e10
        self.out_array /= self.out_count

        # x, weight, bias
        return self.out_array, d_weight, d_bias


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
        self.tensor.output_value = 1 / (1 + np.exp(-x.tensor.output_value))
        # if x.tensor.output_value >= 0:
        #     self.tensor.output_value = 1/(1 + np.exp(-x.tensor.output_value))
        # else:
        #     self.tensor.output_value = np.exp(-x.tensor.output_value) / (1 + np.exp(-x.tensor.output_value))
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
        返回和方差的节点，而不是类本身
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





