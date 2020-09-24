# warrior
 A deep learning framework for learning. Of course it's fun, too

## 勇士深度学习框架
##### (仅供学习，当然如果你想用它来训练我也支持，如果你想清楚的话)
当前版本1.5.0

## 勇士基于什么、有什么要求？

基于numpy</br>
使用起来没什么要求</br>

## 勇士实现了什么

### 1. 拥有静态图和动态图两种模式
静态图隶属 StaticGraph 模块</br>
其拥有上下文管理器Session、节点Operation、图Graph</br>
节点中还有熟悉的nn模块😀（如果你喜欢tensorflow1.x的话)</br>
其中实现的节点并不多（但你可以自己实现一个节点)</br>
其要求就是实现节点的计算和求导</br>

动态图隶属 DynamicGraph 模块</br>
其拥有fluid模块（是的，仿的PaddlePaddle飞桨的库的名字）</br>
推荐一下飞桨，真的让我学到了很多东西</br>
你可以自定义一个自己的动态图模型</br>
其要求就是继承DynamicGraph这个类并实现forward</br>
剩下的交给warrior</br>

##### 同样的，静态图在没开启图会话前tensor流中是没有值的
##### 你可以直接打印warrior.Constant(1) + warrior.Constant(1)，其结果应当是一个Add节点且空有形状没有值

### 2.模型的保存和加载

当然，实现的很简陋，是基于变量的名字来保存和读取</br>
而且仅仅是保存了变量而并非整个模型</br>
个人能力有限，请随意吐槽（但不要让我听到😀）</br>

### 3.自动反向传播以及更多Demo和完善的注释
比如线性回归demo(动态图和静态图)</br>
比如手写数字识别(全连接)demo</br>
注释应当很完善吧(应该吧?)(我会补齐的)</br>


### 4.其他更多，欢迎体验一下
那么？如何安装呢？</br>


## 如何安装勇士？
简单！这里提供whl</br>
如果我没有上传GitHub那么可能是还没来得及</br>
通过 [此链接](https://www.nullius.cn/wp-content/uploads/2020/07/warrior-1.5.0-py3-none-any.whl_.zip) 同样可以通过我的博客下载</br>

cmd进入 warrior-1.5.0-py3-none-any.whl 所在目录（是压缩包的先解压）</br>
随后通过pip指令安装</br>
> pip install warrior-1.5.0-py3-none-any.whl

试一下是否安装成功</br>

```python
import warrior as wr

a = wr.Constant(1, name="a")
b = wr.Constant(2, name="b")

with wr.Session() as sess:
    res = sess.run(a+b)
print(res)

```

## 如何使用勇士?

噢~抱歉,暂时没有文档呢... </br>
不过我相信聪明的你一定能够很快的理解</br>
这也是学习的一部分嘛</br>

### 快速使用
线性回归demo

```python

import warrior as wr
import numpy as np
# 不使用可视化的可以不导入plt
import matplotlib.pyplot as plt

using_restore = True
checkpoint_path = "test.emblem"

# 生成100个-1 到 1之间的等差数列数
input_x = np.linspace(-1, 1, 100)
np.random.shuffle(input_x)

input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5
# 压成（批大小100，1）
input_x = np.reshape(input_x, (-1, 1))
input_y = np.reshape(input_y, (-1, 1))

# 使用全局默认图
with wr.Graph().as_default():
    # 定义占位符，设置形状和类型
    x = wr.Placeholder(shape=input_x.shape, dtype=type(input_x), name="x")
    real_y = wr.Placeholder(input_y.shape, type(input_y), name="real_y")

    weight = wr.Variable(np.random.normal(loc=0.0, scale=1.0, size=(1, 1)), name="weight")
    bias = wr.Variable(np.random.normal(loc=0.0, scale=1.0, size=()), name="bias")

    y = weight*x + bias
    # sse和方差损失(等reduce mean可以变均方差损失,或者sse/批大小）
    sse_loss = wr.ReduceSum(wr.Square(y - real_y))

    # 梯度下降优化器
    optimizer = wr.train.GradientDescentOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(sse_loss)

    with wr.Session() as sess:
        epoch = 30

        if os.path.exists(checkpoint_path) and using_restore:
            sess.restore(checkpoint_path)

        for step in range(epoch):
            loss = sess.run(sse_loss, feed_dict={
                x: input_x, real_y: input_y
            }, noauto_using_fast=True)

            print("step: {}, loss: {}".format(str(step), str(loss)))
            sess.run(train_op)

        sess.save("test.emblem", is_cover=True)
        # 与此相同，默认会自动补全后缀（也支持别的后缀但不建议)
        # sess.save("test", is_cover=True)

        pred_w, pred_b = sess.run(weight, bias, feed_dict={
                x: input_x, real_y: input_y
            }, noauto_using_fast=True)
    print(pred_w, pred_b)
# 可视化代码，没导入库的可以不写
max_x, min_x = np.max(input_x), np.min(input_x)
max_y, min_y = float(pred_w) * max_x + float(pred_b), float(pred_w) * min_x + float(pred_b)

plt.plot([max_x, min_x], [max_y, min_y], color='r')
plt.scatter(input_x, input_y)
plt.show()

```
其余例如动态图demo以及全连接神经网络demo👇</br>
更多请参考我的[博客](https://www.nullius.cn/archives/314)