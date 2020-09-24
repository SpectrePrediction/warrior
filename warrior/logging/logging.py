import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] - \n %(message)s\n')  #, datefmt='%d-%b-%y %H:%M:%S')


class PlaceholderError(Exception):
    pass


class SaveWarning(Exception):
    pass


class SaveError(Exception):
    pass


class WarriorIsSoSad(Exception):
    pass


class Error:
    """
    “勇士” 内置错误类
    封装自logging类中的logging.error
    其拥有独特的参数 is_raise 和修改过的exc_info

    :param is_raise bool类型 为True将会使Error退出程序 默认为False
    :param exc_info 捕获Exception类型的详细错误 仅能使用显式传入 他会使Error带上
    详细数据，包括错误路径和错误行数等。非常建议使用

    """
    def __new__(cls, msg, is_raise=False, *args, **kwargs):
        """
        :param msg: str 抛出的错误信息
        :param is_raise: bool 是否阻断程序
        :param args: 与logging类一致
        :param kwargs: 与logging类一致
        :return: None
        注：exc_info被从原来的logging中修改
            exc_info需要捕获的Exception类型详细数据 仅能使用显式传入 他会使Error带上
            详细数据，包括错误路径和错误行数等。非常建议使用
        举例：
            try：
                raise Exception（'test'）
            except Exception as err：
                Error("msg", exc_info=err)
            print>>
                    [msg]
                    [where]
                    [line] and so on
        """
        exc_info = kwargs.get("exc_info", None)
        if is_raise:
            msg_header_list = ("[ 错误内容: ", "[ 错误类型 ]", '[ 错误路径 ]')
        else:
            msg_header_list = ("[ 警告内容: ", "[ 警告错误类型 ]", '[ 警告路径 ]')
        msg = msg_header_list[0] + msg + " ]"

        if exc_info:
            try:

                msg += "\n " + msg_header_list[1] + ": {}".format(str(type(exc_info)).split("'")[1])+\
                       '\n ' + msg_header_list[2] + ': "' + exc_info.__traceback__.tb_frame.f_globals["__file__"] + \
                       '"\n [ 行数 ]: ' + str(exc_info.__traceback__.tb_lineno)
                kwargs['exc_info'] = False

            except AttributeError:

                if isinstance(exc_info, type):
                    msg += "\n " + msg_header_list[1] + ": {}".format(str(exc_info).split("'")[1])
                    kwargs['exc_info'] = False

        logging.error(msg=msg, *args, **kwargs)
        if is_raise:
            exit(-1)

