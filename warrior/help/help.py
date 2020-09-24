
class Help:
    """
    他和python内置的help一样
    那为什么要大费周章的使用他呢？
    我也好奇
    """
    def __new__(cls, *args, **kwargs):
        help(*args, **kwargs)

