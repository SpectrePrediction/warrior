from .Tensor import *
from ..Graph.graph import *
from ...logging.logging import *
from queue import Queue


class Operation(object):
    """
    操作节点（元类）
    假设函数 x + y = z
    此处x y z + 都应当是一个节点
    以此方便进行梯度计算和反向传播

    操作节点接受Tensor
    Tensor中存放零个或多个节点
    输出一个Tensor，中存放零或多个节点
    """
    def __init__(self, *input_node, name=None):
        # 节点所属的图 (DEFAULT_GRAPH是全局变量)
        self.graph = DEFAULT_GRAPH

        self.name = name + DEFAULT_GRAPH.get_graph_id() if name else DEFAULT_GRAPH.get_op_default_name(self.__class__)

        self.tensor = Tensor(*input_node, dtype=None, name=self.name)

        # 将每一个上一个输入的节点中的下一个节点指向自己
        # 同时检查dtype是否一致，最终得到的dtype将放入此节点的Tensor中
        dtype = None
        for node in input_node:
            tensor = node.tensor
            try:
                if dtype:
                    # 如果已经有存放dtype了，进行比较，不同无法进行后续
                    assert dtype == tensor.dtype
                else:
                    dtype = tensor.dtype
            except AssertionError as err:
                Error("节点类型不一致" + str(tensor) +
                      " is not " + dtype, is_raise=True, exc_info=err)
            tensor.output_nodes.append(self)

        self.tensor.dtype = dtype
        self.graph.operations.append(self)

    def __str__(self):
        return str(self.tensor)

    def __repr__(self):
        return self.__str__()

    def compute_output(self):
        """
        完成此节点的计算
        :return:
        """
        Error("子类没有构建compute_output函数" + str(self.tensor), is_raise=True, exc_info=NotImplementedError)

    def auto_output(self):
        """
        自动完成此节点和之前节点的计算
        尚诺此节点中包含未计算的节点
        将自动调用未技术节点的auto_output
        注：诺此节点的根节点存在占位符，无法使用
        :return:
        """
        Error("子类没有构建auto_output函数" + str(self.tensor), is_raise=True, exc_info=NotImplementedError)

    def compute_gradient(self, grad=None):
        """
        计算此节点的梯度
        :param grad: 一般是此节点的值的形状的1
        :return:
        """
        Error("子类没有构建compute_gradient函数" + str(self.tensor), is_raise=True, exc_info=NotImplementedError)

    def __add__(self, other):
        """
        加法
        :param other: 另一个相加的数
        :return: 加法节点
        """
        return Add(self, other)

    def __neg__(self):
        """
        负号
        :return:返回负号节点
        """
        return Negative(self)

    def __sub__(self, other):
        """
        相减
        :param other:
        :return: 返回加法节点
        """
        return Add(self, Negative(other))

    def __mul__(self, other):
        """
        相乘
        :param other:
        :return: 返回乘法节点
        """
        return Multiply(self, other)


class Constant(Operation):

    def __init__(self, value, name=None):
        if value is None:
            Error("Constant value: None值不被支持", is_raise=True, exc_info=ValueError)

        self.value = value
        # 注意，根节点没有构造父类,名字得自己取
        self.name = name + DEFAULT_GRAPH.get_graph_id() if name else DEFAULT_GRAPH.get_op_default_name(self.__class__)

        self.tensor = Tensor(None, dtype=type(value), name=self.name)
        self.graph = DEFAULT_GRAPH
        self.graph.constants.append(self)

    def compute_output(self):
        if self.tensor.output_value is None:
            self.tensor.output_value = self.value
        return self.tensor.output_value

    def auto_output(self):
        if self.tensor.output_value is None:
            self.tensor.output_value = self.value
        return self.tensor.output_value


class Variable(Operation):
    def __init__(self, initial_value_or_op=None, name=None, trainable=True):
        """
        在tensorflow中，Variable是通过使用函数来更新变量的值。
        我的本意是改变他，使得他可以直接承接节点来完成更新
        但我有些犹豫。也许会在未来版本被移除此特性
        他目前还没有变量的样子，比如改变值
        如果未来修改了他，那么output中的if self.tensor.output_value is None需要移除
        :param initial_value_or_op:
        :param name:
        :param trainable:
        """
        if initial_value_or_op is None:
            Error("Variable initial_value: 必须指定初始值", is_raise=True, exc_info=ValueError)

        self.name = name + DEFAULT_GRAPH.get_graph_id() if name else DEFAULT_GRAPH.get_op_default_name(self.__class__)
        self.initial_value = None

        if isinstance(initial_value_or_op, Operation):
            Error("你之所以会看到此条警告，原因是你使用Variable来承接了一个节点\n"
                  " 警告节点 " + str(self.name) + ", \n其承接节点 " + str(initial_value_or_op.name) + "\n"
                  "Variable的使用我觉得并不应被用来承接其他节点，他更应该是一个数\n"
                  "我还在观察他。当他作为节点变量时，变量的梯度计算并未完成\n"
                  "所以请注意，训练时最好是传入值而非节点，否则可能出现使用训练时无法求梯度错误\n"
                  "当然，非训练下你可以 暂时的！ 快捷的使用他", is_raise=False, exc_info=FutureWarning)
            self.graph = DEFAULT_GRAPH

            self.tensor = Tensor(initial_value_or_op, dtype=initial_value_or_op.tensor.dtype, name=self.name)

            # 将上一个输入的节点中的下一个节点指向自己
            initial_value_or_op.tensor.output_nodes.append(self)

            self.graph.variables.append(self)
            if trainable:
                self.graph.trainable_variables.append(self)
        else:
            self.initial_value = initial_value_or_op

            self.tensor = Tensor(None, dtype=type(initial_value_or_op), name=self.name)

            self.graph = DEFAULT_GRAPH
            self.graph.variables.append(self)
            if trainable:
                self.graph.trainable_variables.append(self)

    def compute_output(self):
        if self.tensor.output_value is None:
            if self.initial_value is not None:
                self.tensor.output_value = self.initial_value
            else:
                try:
                    x, = self.tensor.input_nodes
                    self.tensor.output_value = x.tensor.output_value
                except TypeError as err:
                    Error("出现无法获取节点值，可能源于你未启动图会话"
                          "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        if self.tensor.output_value is None:
            if self.initial_value is not None:
                self.tensor.output_value = self.initial_value
            else:
                try:
                    x, = self.tensor.input_nodes
                    if x.tensor.output_value is None:
                        x.auto_output()
                    self.tensor.output_value = x.tensor.output_value
                except TypeError as err:
                    Error("出现无法获取节点值，可能源于你未启动图会话"
                          "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value


class Placeholder(Operation):
    def __init__(self, shape, dtype, name=None):
        self.value = None
        self.name = name + DEFAULT_GRAPH.get_graph_id() if name else DEFAULT_GRAPH.get_op_default_name(self.__class__)

        self.tensor = Tensor(None, dtype=dtype, name=self.name)
        self.tensor.shape = tuple(shape)

        self.graph = DEFAULT_GRAPH
        self.graph.placeholders.append(self)

    def auto_output(self):
        raise PlaceholderError(self.tensor)


class Add(Operation):

    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        try:
            x, y = self.tensor.input_nodes
            self.tensor.output_value = np.add(x.tensor.output_value, y.tensor.output_value)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, y = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        if y.tensor.output_value is None:
            y.auto_output()
        self.tensor.output_value = np.add(x.tensor.output_value, y.tensor.output_value)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.tensor.output_value for node in self.tensor.input_nodes]

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        grad_wrt_x = grad
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


class Negative(Operation):

    def __init__(self, x, name=None):

        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.tensor.output_value = -x.tensor.output_value
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        self.tensor.output_value = -x.tensor.output_value
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.tensor.output_value)
        return -grad


class Multiply(Operation):
    """
    对应元素相乘
    向量乘矩阵也都可以
    会自动广播
    [[1, 2 , 3]  乘 [1, 2 ,3 ] -> [[1, 4, 9]
     [4, 5, 6]]                    [4, 10, 18]]
    """
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        try:
            x, y = self.tensor.input_nodes
            self.tensor.output_value = np.multiply(x.tensor.output_value, y.tensor.output_value)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, y = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        if y.tensor.output_value is None:
            y.auto_output()
        self.tensor.output_value = np.multiply(x.tensor.output_value, y.tensor.output_value)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        """
        x * y
        dxdy : y+x
        :param grad:
        :return:
        """
        x, y = [node.tensor.output_value for node in self.tensor.input_nodes]

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        grad_wrt_x = grad * y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad * x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


class MatMul(Operation):
    """
    矩阵相乘 x乘
    使用dot而非matmul，暂不知道区别，起码在2维上。
    [[3., 3.]] x [[2.], [2.]] -> [[12.]]
    1*2 x 2*1 -> 1*1
    """
    def __init__(self, x, y, name=None):

        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        try:
            x, y = self.tensor.input_nodes
            self.tensor.output_value = np.dot(x.tensor.output_value, y.tensor.output_value)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, y = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        if y.tensor.output_value is None:
            y.auto_output()
        self.tensor.output_value = np.dot(x.tensor.output_value, y.tensor.output_value)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.tensor.output_value for node in self.tensor.input_nodes]

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        dfdx = np.dot(grad, np.transpose(y))
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]


class Log(Operation):
    """
    求对数
    """
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)
        # print(self.tensor.input_nodes[0])
        # print(self.tensor.input_nodes)

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.tensor.output_value = np.log(x.tensor.output_value)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        self.tensor.output_value = np.log(x.tensor.output_value)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):

        x = self.tensor.input_nodes[0].tensor.output_value
        if grad is None:
            grad = np.ones_like(self.tensor.output_value)
        return grad*1/x


class ReduceSum(Operation):
    """
    累加求和
    等同于tf.reduce_sum
    [[1, 1, 1]
     [1, 1, 1]] -> 6
     axis:
      0 -> [2, 2, 2]
      1 -> [3, 3]
      1,keepdims=True -> [[3], [3]]
      [0,1] -> 6
    """
    def __init__(self, x, axis=None, keepdims=np._NoValue):

        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.tensor.output_value = np.sum(x.tensor.output_value, self.axis, keepdims=self.keepdims)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        self.tensor.output_value = np.sum(x.tensor.output_value, self.axis, keepdims=self.keepdims)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        input_value = self.tensor.input_nodes[0].tensor.output_value

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


class Square(Operation):
    """
    平方
    """
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.tensor.output_value = np.square(x.tensor.output_value)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()
        self.tensor.output_value = np.square(x.tensor.output_value)
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        input_value = self.tensor.input_nodes[0].tensor.output_value

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        return grad*np.multiply(2.0, input_value)


class Reshape(Operation):
    def __init__(self, x, shape, name=None):
        super(self.__class__, self).__init__(x, name=name)
        self.old_shape = None
        self.shape = shape

    def compute_output(self):
        try:
            x, = self.tensor.input_nodes
            self.old_shape = x.tensor.output_value.shape
            self.tensor.output_value = np.reshape(x.tensor.output_value, self.shape)
        except TypeError as err:
            Error("出现无法获取节点值，可能源于你未启动图会话"
                  "或者在图会话中没有提供占位符具体的值" + str(self.tensor), is_raise=True, exc_info=err)
        return self.tensor.output_value

    def auto_output(self):
        x, = self.tensor.input_nodes
        if x.tensor.output_value is None:
            x.auto_output()

        self.compute_output()
        return self.tensor.output_value

    def compute_gradient(self, grad=None):
        # input_value = self.tensor.input_nodes[0].tensor.output_value

        if grad is None:
            grad = np.ones_like(self.tensor.output_value)

        return np.reshape(grad, self.old_shape)


def compute_gradients(target_op):
    grad_table = dict()

    grad_table[target_op] = np.ones_like(target_op.tensor.output_value)

    queue = Queue()
    queue.put(target_op)

    visited = set()
    visited.add(target_op)
    compute_gradient_cache = {}

    while not queue.empty():
        node = queue.get()
        # print(node)

        if node != target_op:
            grads_wrt_node_output = []

            for output_node in node.tensor.output_nodes:

                # grad_wrt_output_node_output = grad_table[output_node]
                grad_wrt_output_node_output = grad_table.get(output_node, None)

                if grad_wrt_output_node_output is None:
                    continue

                grad_wrt_node_output, cache_num = compute_gradient_cache.get(output_node, (None, None))

                if grad_wrt_node_output is None or cache_num < 1:
                    # 不满足缓存条件
                    grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                else:
                    compute_gradient_cache[output_node][1] -= 1

                    # if compute_gradient_cache[output_node][1] < 1:
                    #     del compute_gradient_cache[output_node]

                if len(output_node.tensor.input_nodes) > 1:
                    # print(grad_wrt_node_output)
                    if cache_num is None or cache_num < 0:
                        compute_gradient_cache[output_node] = [grad_wrt_node_output,
                                                               output_node.tensor.input_nodes.__len__() - 1]

                    input_node_index = output_node.tensor.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        if hasattr(node.tensor, 'input_nodes'):
            for input_node in node.tensor.input_nodes:
                if input_node not in visited:
                    if input_node:
                        visited.add(input_node)
                        queue.put(input_node)

    return grad_table


if __name__ == '__main__':
    pass
    # print(Operation())
