import numpy as np
from ..logging.logging import *


class DynamicGraph(object):

    def __init__(self):
        pass

    def forward(self, *graph_input):
        Error("自定义模型没有定义forward", exc_info=RuntimeError, is_raise=True)

    def __call__(self, *graph_input, **kwargs):
        if kwargs:
            Error("动态图暂不支持使用显示传参")
            return

        if len(graph_input) == 1:
            try:
                out = self.forward(*graph_input)
                return out
            except TypeError as err:
                Error("自定义模型参数数量定义错误", exc_info=err, is_raise=True)
            except Exception as err:
                Error(str(err), exc_info=err, is_raise=True)
        else:
            Error("暂时的。传入参数数量只支持为1,但传入" + str(len(graph_input)), exc_info=FutureWarning)

