import tensorflow as tf 

#定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0,2.0,3.0,4.0],name="input1")
input2 = tf.get_variable(name="input2",initializer=([3.,2.,1.,0.0]))
output = tf.add_n([input1,input2],name="add")

#生成一个写日志的writer，并将当前的TF计算图写入日志。TF提供了多种写
#日志文件的API，
writer = tf.summary.FileWriter("./path/to/log",tf.get_default_graph())
writer.close()
