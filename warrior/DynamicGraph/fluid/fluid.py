from ...StaticGraph.Operation.Operation import *
from ...StaticGraph.Operation.train.train import *
from ...StaticGraph.Operation.nn.nn import *
from ...StaticGraph.Session.Session import *
from ...StaticGraph.Graph.graph import *
from ..DynamicGraph import *


class DynamicBackward:

    def __init__(self):
        self.sess = Session()

    def step_init(self, train_op):
        try:
            assert isinstance(train_op, Operation)
        except AssertionError as err:
            Error("{}的optimizer类型错误\n 应当是：{}, 但传入{}".
                  format(str(self), "train_op(Operation)", str(type(train_op))),
                  is_raise=True, exc_info=err)

        self.train_op = train_op
        self.loss = train_op.loss
        return self

    def step(self, is_using_exec=False):
        if is_using_exec:
            return self.sess.exec_run(self.loss, self.train_op)[0]
        else:
            return self.sess.run(self.loss, self.train_op)[0]

    def save_model(self, checkpoint_path, is_cover=False, write_step=None):
        return self.sess.save(checkpoint_path, is_cover, write_step)

    def load_model(self, checkpoint_path):
        return self.sess.restore(checkpoint_path)


def from_numpy(numpy_input, name=None):
    temp = Constant(numpy_input, name=name)
    temp.auto_output()
    return temp


class Linear(DynamicGraph):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Variable(np.random.normal(loc=0.0, scale=1.0, size=(1, 1)), name="weight")
        self.bias = Variable(np.random.normal(loc=0.0, scale=1.0, size=(1, 1)), name="bias") if bias else None

    def forward(self, graph_input):
        # graph_input is a Constant op
        try:
            assert isinstance(graph_input, Constant)
        except AssertionError as err:
            Error("{}定义后应当传入节点. 但传入 {},也许你需要对数据使用from_numpy".format(self, type(graph_input)), is_raise=True, exc_info=err)

        for axis, size in enumerate(np.shape(graph_input)):
            try:
                assert size == self.in_features
            except AssertionError as err:
                Error("{}的维度不对应\n 应当是：{}, 但传入{}".
                      format(str(self), str(self.in_features),
                             str(size)),
                      is_raise=True, exc_info=err)

        pred_y = Multiply(self.weight, graph_input)
        # pred_y = MatMul(self.weight, graph_input)
        if self.bias:
            pred_y += self.bias

        return pred_y









