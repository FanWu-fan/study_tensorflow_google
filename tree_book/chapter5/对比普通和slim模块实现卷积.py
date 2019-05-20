import tensorflow as tf 

#直接使用TF原始API实现卷积层。
with tf.variable_scope(scope_name):
    weights = tf.get_variable("weights",...)
    biases = tf.get_variable("bias",...)
    conv = tf.nn.conv2d(...)
relu = tf.nn.relu(tf.nn.bias_add(conv,biases))



#使用TF-Slim实现卷积层。通过TF-Slim可以在一行中实现一个卷积层的
#前向传播算法。slim.conv2d函数的有3个参数是必填的。第一个参数为输入节点
#矩阵，第二参数是当前卷积层过滤器的深度，第三个参数是过滤器的尺寸
#可选的参数有过滤器移动的步长、是否使用全0填充、激活函数的选择以及
#变量的命名空间等。
net = slim.conv2d(input,32,[3,31])
