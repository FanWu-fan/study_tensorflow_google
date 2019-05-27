import tensorflow as tf 

#将输入定义放入各自的命名空间中，
#从而使得TB可以根据命名空间来整理可视化效果图上的节点
with tf.variable_scope("input1"):
    input1 = tf.constant([1.0,2.,3.],name = "input1")
with tf.variable_scope("input2"):
    input2 = tf.get_variable("input2",
    initializer=tf.random_normal([3]))
output = tf.add_n([input1,input2],name="add")

writer = tf.summary.FileWriter("./path/to/log",tf.get_default_graph())
writer.close()