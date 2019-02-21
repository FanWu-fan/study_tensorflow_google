# Chapter1 计算图介绍
# Tensorflow
## 深度学习的历史
神经网络的发展可以分为三个阶段：  
* 早期提出单个神经元结构，简单的线性加权和来模拟这个变换，n个输入值提供给McCull-Pitts 结构，经过加权和一个阈值函数得到0或1的输出。（但是权重学习人手工设置）感知机模型，首个根据样例数据来学习特征权重的模型。
* 分布式知识表达和神经网络的反向传播算法。分布式知识表达的核心思想是现实世界的知识和概念应该通过多个神经元(neuron)来表达，比如识别车，有n种颜色，m种车。如果一个神经元对应一种组合，那么需要$n*m$种组合，而一些神经元表达颜色，一些神经元表达车型，那么要$n+m$组合，而且可以推广到新的概念，神经网络由宽度走向了深度。很好的解决类似亦或等线性不可分的问题。

##深度学习开源工具总结表
工具名称 | 主要维护人员 | 
:--: | :--:
Caffe | 加州大学伯克利分校视觉与学习中心
Deeplearning4j | Skymind
CNTK | 微软研究院
MXNet | 分布式机器学习社区
PaddlePaddle | 百度
Torch | Fackbook

## Tensorflow环境搭建
Tensorflow依赖的两个最主要的工具包：**Prorocol Buffer**和**Bazel**.

### Protocol Buffer
**Protocol Buffer** 是谷歌开发的处理结构化数据的工具。
```protobuf
name:张三
id:12345
email: zhangsan@ad.com
``` 
这里的结构化数据和大数据中的结构化数据的概念不同，本节介绍的结构化数据是指 拥有多种属性的数据。比如用户信息有 名字、ID、E-mail三种不同的属性，那么它就是一个结构化数据。
当要传输结构化数据时，先要序列化，即 将结构化的数据变成数据流的形式，就是变为一个字符串。 如何将结构化的数据序列化，并且从序列化后的数据流换原出原来的结构化数据，统称为 处理结构化数据。
除了**Protocol Buffer**之外，**XML**和 **JSON**是两种比较常用的结构化数据处理工具。
**XML**
```xml
<user>
    <name>张三</name>
    <id>12345</id>
    <email>zhangsan@abc.com</email>
</user>
```
**JSON**
```json
{
    "name": "张三“,
    "id": "12345",
    "email": "zhangsan@ad.com",
}
```
 * **Protocol Buffer**序列化后的数据不可读，是二进制流，不同于 **XML** 和 **JSON** 
 *  **XML** 和 **JSON**的数据信息都包含在序列化之后的数据中，不需要任何其他信息就能换源序列化之后的数据，但是使用 **Protocol Buffer**时需要先定义数据的格式。还原一个序列化之后的数据将需要使用到这个定义好的数据格式。所以要比  **XML** 和 **JSON**快 20 到 100倍，数据小 3到10倍。
```
message user{
    optional string name = 1; //可选的
    required int32 id = 2;      //必须的
    repeated string emial = 3;  //可重复的，可以使用列表
}
```
**Protocol Buffer** 定义的数据格式文件一般保存在.proto文件中。每个message代表了一类结构化的数据，比如这里的用户信息，message里面定义了每一个属性的类型和名字。Protocol Buffer里属性的类型可以时像布尔型、整数型、实数型、字符型这样的基本类型，也可以是另外一个message。
在message中，定义一个属性是 必须的(**required**)[*所有的实例都需要有这个属性*], 可选的(**optional**)[*属性的取值可以为空*],可重复的(**repeated**)[*属性的取值可以是一个列表*],
分布式 **Tensorflow** 的通信协议 gRPC 也是以 Protocol Buffer作为基础。

### Bazel

Bazel是谷歌开源的自动化构建工具，相比于传统的 Makefile,Ant或者Maven,Bazel在速度、可伸缩性、灵活性更加出色。项目空间(workspace)，这个文件夹包括了编译一个软件的所需要的源代码以及输出编译结果的软连接(symbolic link)地址。根目录需要一个WORKSPACE文件，此文件定义了对外部资源的依赖关系。
在一个项目空间内，Bazel通过BUILD文件来寻找需要编译的目标，BUILD文件采用类似python的语法来指定每一个编译目标的输入、输出以及编译方式。与Makefile这种开放式的编译工具不同，Bazel的编译方式是事先定义好的，BAzel对python支持的编译方式有：py_binary,py_libary和 py_test。
* **py_binary** 将python程序编译为可执行文件
* **py_test** 编译python测试程序
* **py_libary** 将python程序编译成库函数供其他的py_binary或py_test调用。

如下所示，在样例空间中有4个文件：
WORKSPACE, BUILD, hello_main.py 和 hello_lib.py
WORKSPACE给出此项目的外部依赖关系，为了简单起见，这里使用一个空文件，表明这个项目没有对外部的依赖。hello_lib.py完成打印“hello world”的简单功能，它的代码如下：
```python
def print_hello_world():
    print("Hello World")
```
hello_main.py通过调用hello_lib.py中定义的函数来完成输出，它的代码如下：
```python
import hello_lib
hello_lib.print_hello_world()
```
在BUILD文件中定义了两个编译目标：
```
py_library(
    name = "hello_lib",
    srcs = [
        "hello_lib.py",
    ]
)

py_binary(
    name = "hello_main",
    srcs = [
        "hello_main.py",
    ],
    deps = [
        ":hello_lib",
    ],
)
```
BUILD文件由一系列编译目标组成的，**定义编译目标的先后顺序不会影响编译的结果**，在每一个编译目标的第一行指定编译的方式(py_library,py_binary)。在每一个编译目标中的主体需要给出编译的具体信息，编译的具体信息是通过定义 name, srcs, deps 等属性完成的。
* **name**: 编译目标的名字，这个名字用来指代这一条编译目标。
* **src**: 给出编译所需要的源代码，这一项可以是一个列表。
* **deps**: 给出了编译所需要的依赖关系，比如hello_main.py 需要调用hello_lib.py中的函数，所以hello_main的编译目标将hello_lib作为依赖关系。在这个项目空间中运行编译操作 bazel build:hello_main

## Tensorflow计算模型--计算图
Tensorflow的名字当中已经说明了它最重要的两个概念--Tensor和Flow。Tensor就是张量，可以理解为多维数组，体现了数据结构，那么Flow体现了计算模型，Tensorflow是通过计算图的形式来表述计算的编程系统。每一个计算都是计算图上的节点，而节点的边描述了计算之间的依赖关系。
Tensorflow程序一般可以分为两个阶段，第一个定义计算图中的所有计算，其中为了建模的方便，TF会将常量转化成一种永远输出固定值的运算。
```python
import tensorflow as tf 
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")
result = a + b
```
系统会自动维护一个默认的计算图，通过tf.get_default_graph()函数可以获取当前默认的计算图，以下代码示意了如何获取默认计算图以及如何查看一个运算所属的计算图。
```python
print(a.graph is tf.get_default_graph())
True
```
除了使用默认的计算图，TF支持通过**tf.Graph**函数来**生成新的计算图**，不同计算图上的张量和运算都不会共享。以下代码示意了如何在不同计算图上定义和使用变量。
```python
import tensorflow as tf
"""
定义位置： tensorflow/python/ops/variable_scope.py
此上下文管理器验证(可选)指来之同一个计算图，确保计算图是默认计算图，并且推送名称范围和变量范围。

如果name_orscope不是None，它按照原样使用，如果name_or _scope是None,则使用default_name,
在这种情况下，如果先前在同一范围内同一名字已经被使用，它将会被附加_N成为唯一的名字

variable_scope允许你创造一个新的变量并共享已经创建的变量，同时提供不会意外的创建或者共享的检查，
"""
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量“V”，并设置初始值为0
    v = tf.get_variable(
        name="v", shape=[1],initializer = tf.zeros_initializer
    )
    #这里的initializer是名词，代表的是设置属性、状态
g2 = tf.Graph()
with g2.as_default():
    #在计算图g2中定义变量"v",并且设置初始值为1
    v = tf.get_variable(
        "v",shape=[1], initializer=tf.ones_initializer
    )

#在计算图g1中读取变量"v"的取值。
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    #这里的initialize是动词，执行初始化所有变量的操作
    with tf.variable_scope("",reuse = True): #定义变量
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    #这里的initialize是动词，执行初始化所有变量的操作
    with tf.variable_scope("",reuse = True):
        print(sess.run(tf.get_variable("v")))
```
上面的代码产生了两个计算图，每个计算图中定义了一个名字为"v"的变量，在计算图g1中，v初始化为0；g2中，v初始化为1.
TF中计算图不仅仅可以用来隔离张量和计算，还提供管理张量和计算的机制。计算图可以通过tf.Grapg.device函数来指定运行的设别。
```python
g = tf.Graph()
with g.device('/gpu:0'):
    result = a +b
```
在一个计算图中，可以通过集合(collection)来管理不同类别的资源，比如通过tf.add_to_collection函数可以将资源加入一个或多个集合中，然后通过tf.get_collection获取一个集合里面的所有资源，这里的资源可以是张量，变量或者运行TF程度所需要的队列资源。同时TF也自动管理了一些最常用的集合。
集合名称 | 集合内容 | 使用场景
:--: | :--: | :--:|
tf.GraphKeys.VARIABLES|所有变量 |持久化TF模型
tf.GraphKeys.TRAINABLE_VARIABLES | 可学习的变量(神经网络的参数)|模型训练、生成模型可视化内容
tf.GraphKeys.SUMMARIES | 日志生成相关的张量 |TF的计算可视化
tf.GraphKeys.QUEUE_RUNNERS | 处理输入的QueueRunner | 输入处理
tf.GraphKeys.MOVING_AVERAGE_VARIABLES | 所有计算了滑动平均值的变量 | 计算了变量的滑动平均值


