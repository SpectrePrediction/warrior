from ...logging.logging import *
from ..Operation.Operation import *
import os
import json


class Session(object):

    def __init__(self):
        # 开启会话获取当前使用图
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        # 启动上下文管理器
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时自动释放会话
        self.close()

    def get_all_nodes(self):
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations)
        #  + self.graph.trainable_variables) 舍弃
        return all_nodes

    def save(self, checkpoint_path, is_cover=False, write_step=None):
        """
        Their shields bear the device of the Blazing Sun.
        Look, do you know what that emblem means?
        他们的盾形徽章上有炽阳的图案。
        看，你知道那个徽章表示什么吗？
        so emblem endswith
        :param checkpoint_path: 以此结尾emblem，但允许输入其他结尾，默认结尾为emblem
        :return:
        """
        if not checkpoint_path.endswith(".emblem"):
            other_end = checkpoint_path.split('.')[-1]

            if checkpoint_path.split('.')[0] == other_end:
                checkpoint_path += ".emblem"
            else:
                Error("结尾的文件名为 {} ，为什么不是emblem呢?\n但我们尊重你的选则".format(other_end), exc_info=WarriorIsSoSad)

        if write_step:
            checkpoint_path = checkpoint_path.replace(".", "-" + str(write_step) + '.')

        if os.path.exists(checkpoint_path):
            if is_cover:
                Error("已经存在 {} 文件\n但由于选择了if_cover，他将覆盖".format(checkpoint_path), exc_info=SaveWarning)
            else:

                Error("已经存在 {} 文件\n但勇士的徽章依旧会被保存\n太阳的形状\n"
                      "现在他被重新命名(添加后缀)".format(checkpoint_path), exc_info=SaveWarning)
                while os.path.exists(checkpoint_path):
                    checkpoint_new_path = checkpoint_path.split('.')[0] + "_again"
                    checkpoint_path = checkpoint_new_path + '.' + checkpoint_path.split('.')[-1]

        save_dict = dict()
        for trainable_op in self.graph.trainable_variables:
            # 这里仅适配当变量shape为(1, 1)的时候,我在尝试改进（虽然理论上约束也是可行的）
            # 已经修复 v1.1.8 现在标量也支持
            # save_dict[str(trainable_op)] = [list(list(trainable_op.tensor.output_value)[0])]
            # axis_list = None

            shape = trainable_op.tensor.get_shape()
            if shape is not None:
                try:
                    if len(shape) == 0:
                        axis_list = trainable_op.tensor.output_value

                    elif len(shape) == 1:
                        # axis = shape[0]
                        for i in range(shape[0]):
                            axis_list = list(trainable_op.tensor.output_value)

                    elif len(shape) == 2:
                        axis_list = list(trainable_op.tensor.output_value)

                        for s in range(len(axis_list)):
                            axis_list[s] = list(axis_list[s])

                except Exception as err:
                    Error("发生意料之外的错误", is_raise=True, exc_info=err)

            else:
                Error("存在变量无法获取shape！或者shape为None \n{}\n"
                      "可能原因: 没用启动图会话 或者shape无法获取\n"
                      "如果此变量并非是必须的可训练的，请设置\n"
                      "trainable=False, 此举将不会保存此变量".format(trainable_op), is_raise=True, exc_info=SaveError)

            save_dict[trainable_op.tensor.get_save_information()] = axis_list

        with open(checkpoint_path, "w") as f:
            f.write(json.dumps(save_dict))
        # json.loads(json_str)

    def restore(self, checkpoint_path, by_name=True):
        if not checkpoint_path.endswith(".emblem"):
            if checkpoint_path.split('.')[0] == checkpoint_path.split('.')[-1]:
                checkpoint_path += ".emblem"

        try:
            with open(checkpoint_path, "r") as f:
                json_str = f.read()

            op_dict = json.loads(json_str)
        except json.decoder.JSONDecodeError:
            Error("模型损坏！无法读取有效内容\n"
                  "请勿修改模型中的内容", is_raise=True, exc_info=json.decoder.JSONDecodeError)
        except UnicodeDecodeError:
            Error("这个文件遇到无法解码的内容，他是否是模型？", is_raise=True, exc_info=UnicodeDecodeError)
        except FileNotFoundError:
            Error("提供路径下没有模型！", is_raise=True, exc_info=FileNotFoundError)

        for key, value in op_dict.items():
            name = key.split("'")[1].split(' ')[0]
            dtype = key.split("'")[3]

            for trainable_op in self.graph.trainable_variables:
                if trainable_op.tensor.name.split(' ')[0] == name and trainable_op.tensor.dtype.split("'")[1] == dtype:
                    trainable_op.tensor.output_value = np.array(value)
                    break
            else:
                Error("restore加载变量时 \n 变量名 {} 没有找到对应Tensor\n"
                      "可能原因是没有存在此变量或原变量变量名被改变\n"
                      "请检查是否存在 变量名[{}] 类型 [{}] 其值为 {} 的变量".format(
                        name, name, dtype, np.array(value)
                      ), is_raise=not by_name, exc_info=ReStoreWarning if by_name else ReStoreError)

        for trainable_op in self.graph.trainable_variables:
            if trainable_op.tensor.output_value is None:
                Error(" 存在可训练变量还未获取到值\n"
                      " 变量名 {}\n"
                      "如果你无法保证模型与保存前一致，可选择by_name=True\n"
                      "或者将此变量设置为不可训练trainable=False".format(
                        trainable_op.tensor.name.split(' ')[0]
                      ), is_raise=not by_name, exc_info=ReStoreWarning if by_name else ReStoreError)

    def close(self):
        # 将所有节点中的值清空
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.tensor.output_value = None

    def exec_run(self, *operation, feed_dict=None, using_auto=True, noauto_using_fast=True):
        """
        使用compile将字符串转换成字节码，也许速度能更快？也许不能
        他的存在似乎不影响，你可以个凭喜好
        :param operation:一个或多个节点
        :param feed_dict: 占位符数据
        :type feed_dict: Dict
        :param using_auto: 是否使用自动前向，在存在占位符上时，请选False
        :param noauto_using_fast: 在不使用using_auto时，此选项生效，这将影响图遍历的算法选择
        :return: 一个或多个节点的返回
        """

        prog = '''
global outlist''' + '''
def get_node(operation, using_fast):
    graph_nodes = []
    # op = operation

    if not using_fast:
        # temp = op.tensor.input_nodes
        # graph_nodes.insert(0, temp)
        # for input_nodes in temp:
        #     temp = input_nodes.tensor.input_nodes
        #     for i in temp:
        #         if i:
        #             graph_nodes.insert(0, i)
        Error("暂时禁止使用sess.run中参数(noauto_using_fast=False), 将在后续版本开放",
              is_raise=True, exc_info=RuntimeWarning)
    else:
        def postorder_traverse(operation):
            if operation:
                for input_node in operation.tensor.input_nodes:
                    postorder_traverse(input_node)
                graph_nodes.append(operation)

        postorder_traverse(operation)

    return graph_nodes
''' + '''
def run(*operation, feed_dict=None, using_auto=True, noauto_using_fast=True, get_node=None):
    output_list = []
    if feed_dict:
        using_auto = False
        
    if using_auto:
        for op in operation:
            try:
                op.auto_output()
            except PlaceholderError as err:
                Error("存在占位符不应当使using_auto为True: " + str(err), is_raise=False, exc_info=err)
                Error("已自动切换using_auto为False", is_raise=False)

                output_list = []
                using_auto = False
                break
            output_list.append(op.tensor.output_value)

    if not using_auto:
        for op in operation:
            graph_node = get_node(op, noauto_using_fast)
            for node in graph_node:
                if type(node) is Placeholder:
                    try:

                        placeholder_value = feed_dict.get(node, None)

                        # 检查feed_dict是否此占位符的值
                        if placeholder_value is None:
                            Error("没有在feed_dict参数中获取到占位符{}的值".format(str(node)),
                                  is_raise=True, exc_info=PlaceholderError)

                        # check一下是否和此占位符维度和类型对应
                        if str(type(placeholder_value)) != node.tensor.dtype:
                            Error("占位符{}的数据类型不对应 应当是：{}, 但传入{}".format(str(node),
                                  node.tensor.dtype, str(type(placeholder_value))),
                                  is_raise=True, exc_info=PlaceholderError)

                        if np.shape(placeholder_value) != node.tensor.shape:
                            Error("占位符{}的维度不对应 应当是：{}, 但传入{}".
                                  format(str(node), str(node.tensor.shape),
                                        str(np.shape(placeholder_value))),
                                  is_raise=True, exc_info=PlaceholderError)

                        node.tensor.output_value = placeholder_value

                    except AttributeError as err:

                        Error("存在占位符但没有传递feed_dict参数",
                              is_raise=True, exc_info=err)

                else:
                    node.compute_output()

            output_list.append(op.tensor.output_value)
    if output_list.__len__() == 1:
        output_list = output_list[0]
    return output_list
''' + '''
outlist = run(*operation, feed_dict=feed_dict, using_auto=using_auto, noauto_using_fast=noauto_using_fast, get_node=get_node)

'''

        y = compile(prog, '', 'exec')
        exec(y)
        return outlist

    def run(self, *operation, feed_dict=None, using_auto=True, noauto_using_fast=True):
        """
        启动图会话
        :param operation:一个或多个节点
        :param feed_dict: 占位符数据
        :type feed_dict: Dict
        :param using_auto: 是否使用自动前向，在存在占位符上时，请选False
        :param noauto_using_fast: 在不使用using_auto时，此选项生效，这将影响图遍历的算法选择
        :return: 一个或多个节点的返回
        """
        output_list = []
        if feed_dict:
            using_auto = False

        if using_auto:
            for op in operation:
                try:
                    op.auto_output()
                except PlaceholderError as err:
                    Error("存在占位符不应当使using_auto为True: " + str(err), is_raise=False, exc_info=err)
                    Error("已自动切换using_auto为False", is_raise=False)

                    output_list = []
                    using_auto = False
                    break
                output_list.append(op.tensor.output_value)

        if not using_auto:
            for op in operation:
                graph_node = self.get_graph_node(op, noauto_using_fast)

                for node in graph_node:
                    if type(node) is Placeholder:
                        try:
                            placeholder_value = feed_dict.get(node, None)

                            # 检查feed_dict是否此占位符的值
                            if placeholder_value is None:
                                Error("没有在feed_dict参数中获取到占位符{}的值".format(str(node)),
                                      is_raise=True, exc_info=PlaceholderError)

                            # check一下是否和此占位符维度和类型对应
                            if str(type(placeholder_value)) != node.tensor.dtype:
                                Error("占位符{}的数据类型不对应\n 应当是：{}, 但传入{}".format(str(node),
                                      node.tensor.dtype, str(type(placeholder_value))),
                                      is_raise=True, exc_info=PlaceholderError)

                            if np.shape(placeholder_value) != node.tensor.shape:
                                shape = node.tensor.shape
                                for i in range(len(node.tensor.shape)):

                                    if shape[i] is None:
                                        continue
                                    # zyt贡献bug一个，恭喜
                                    elif shape[i] != np.shape(placeholder_value)[i]:
                                        Error("占位符{}的维度不对应\n 应当是：{}, 但传入{},其维度中第 {} 位对应不上".
                                              format(str(node), str(node.tensor.shape),
                                                     str(np.shape(placeholder_value)), str(i)),
                                              is_raise=True, exc_info=PlaceholderError)
                                # Error("占位符{}的维度存在None警告，但通过维度检测".
                                #       format(str(node), is_raise=False, exc_info=PlaceholderError))

                            node.tensor.output_value = placeholder_value

                        except AttributeError as err:

                            Error("存在占位符但没有传递feed_dict参数",
                                  is_raise=True, exc_info=err)

                    else:
                        node.compute_output()

                output_list.append(op.tensor.output_value)
        if output_list.__len__() == 1:
            output_list = output_list[0]
        return output_list

    @staticmethod
    def get_graph_node(operation, using_fast=True):
        """
        得到图流程
        :return: list 从开始到此节点的所有节点
        """
        graph_nodes = []
        # op = operation

        if not using_fast:
            # temp = op.tensor.input_nodes
            # graph_nodes.insert(0, temp)
            # for input_nodes in temp:
            #     temp = input_nodes.tensor.input_nodes
            #     for i in temp:
            #         if i:
            #             graph_nodes.insert(0, i)
            Error("暂时禁止使用sess.run中参数(noauto_using_fast=False), 将在后续版本开放",
                  is_raise=True, exc_info=RuntimeWarning)
        else:
            def postorder_traverse(operation):
                if operation:
                    for input_node in operation.tensor.input_nodes:
                        postorder_traverse(input_node)
                    graph_nodes.append(operation)

            postorder_traverse(operation)

        return graph_nodes



