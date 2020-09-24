import builtins
from ...logging.logging import *


class Graph(object):

    def __init__(self):
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainable_variables, self.op_default_name = [], [], {}

    def __enter__(self):
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = self.old_graph

    def get_op_default_name(self, class_name):
        """
        得到操作节点默认名字
        :param class_name: class 操作节点类名
        :return: 默认名字，其规律满足（操作节点类名+第几个数）
        例子：此图中第一个Constant节点将返回 "1 in graph: 0xFFFFFF"
            而诺传入第二个Constant节点，此时返回 "2 in graph: 0xFFFFF"
            其中数为此图中出现的次数，图地址为此图地址
            因此不同图中，Constant的次数并不会累积
        """
        default_name_num = self.op_default_name.get(class_name, None)
        if default_name_num:
            self.op_default_name[class_name] = default_name_num + 1
        else:
            self.op_default_name[class_name] = 1
        return str(self.op_default_name.get(class_name, None)) + \
               " in graph: " + str(self).split(' ')[-1].split('>')[0]

    def as_default(self):
        """
        使用默认图
        :return:全局默认图，诺没有，则此图

        """
        # return self
        if DEFAULT_GRAPH:
            return DEFAULT_GRAPH
        else:
            return self

    def change_default_graph(self):
        """
        他可以在不使用上下文管理器的情况下更换全局默认图
        作用将使得全局默认图被更换
        他没有上下文管理器的退出机制
        诺在上下文管理器中使用效果和不使用一样
        但他有一定危险性和未知性，暂时禁止使用
        直到我确定这是一个好方法
        """
        Error("暂时禁止使用此函数(change_default_graph), 将在后续版本开放",
              is_raise=True, exc_info=RuntimeWarning)
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = self
        return self


if __name__ == '__main__':
    pass
