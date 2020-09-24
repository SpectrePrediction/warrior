from ..Operation import *


class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        Error("子类没有构建minimize函数", is_raise=True, exc_info=NotImplementedError)


class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate):
        super(self.__class__, self).__init__(learning_rate)

    def minimize(self, loss):

        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def __init__(self):
                super(self.__class__, self).__init__()
                self.loss = loss

            def compute_output(self):
                grad_table = compute_gradients(loss)

                grad = 1
                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    var.tensor.output_value -= learning_rate*grad

            def auto_output(self):
                grad_table = compute_gradients(loss)

                grad = 1
                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    if var.tensor.output_value is None:
                        var.auto_output()
                    var.tensor.output_value -= learning_rate * grad

        return MinimizationOperation()
