import tensorflow as tf 
from tensorflow.python.framework import graph_util

v1 = tf.get_variable(
    "v1",[1],initializer=tf.constant_initializer(1.0))
v2 = tf.get_variable(
    "v2",[1],initializer=tf.constant_initializer(2.0)
)
result = v1+v2

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    #导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层
    #到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    #将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉。一些系统运算
    #也会被转化为计算图中的节点(比如变量初始化操作)，如果只关系程序中定义的
    #某些计算时，和这些计算无关的节点就没有必要导出并保存了。在下面的代码中，
    #最后一个参数['add']给出了需要保存的节点名称，add节点时上面定义的两个
    #变量相加的操作。这一这里给出的是计算节点的名称，所以没有后面的：0
    #有：0，表示的是某个计算节点的第一个输出。而计算节点本身的名称后是没有：0的

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,graph_def,['add']
    )
    #将导出的模型存入文件
    with tf.gfile.GFile("./combined_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())


#####################################################################


with tf.Session() as sess:
    model_filename = "./combined_model.pb"
    #读取保存的模型文件，并将文件解析成对应得 GraphDef Prorocol Buffer
    with tf.gfile.GFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    #将graph_def中保存得图加载到当前的图中，return_elements=['add:0']给出了
    #返回的张量的名称，在保存时给出的是 计算节点的名称，所以为 add.在加载的时候
    #给出的是张量的名称，所以是add:0
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))



