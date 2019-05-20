import tensorflow as tf 
# 通过tf.get_variable的方式创建过滤器的权重变量和偏置项变量，上面介绍了卷积层的
#参数个数只和过滤器的尺寸、深度、以及当前层节点矩阵的深度有关，所以这里声明的参数变量
#是一个四维矩阵，前面两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示
#过滤器的深度。
filter_weight = tf.get_variable(
    "weigths",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1)
)

#和卷积的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一程深度个不同的偏置
#项。本代码中16为过滤器的深度，也是神经网络中下一层节点矩阵的深度
biases = tf.get_variable(
    "biases",[16],initializer=tf.constant_initializer(0.1)
)

#tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法，这个函数的第一个输入为
#当前层的节点矩阵，注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一维对应一个
#输入batch,比如在输入层，input[0,:,:,:]表示第一张图片，input[1,:,:,:]表示第二张图片，以此类推
# [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]。

#tf.nn.conv2d第二个参数提供了卷积层的权重，[filter_height, filter_width, in_channels, out_channels]
# 第三维in_channels，就是参数input的第四维

# 第三个参数维不同维度上的步长，第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数值要求一定是1.
# 这是因为卷积层的步长只对矩阵的长和宽有效，最后一个参数
#是填充(padding)的方法，TF提供SAME或VALID两种选择。SAME表示添加全0，VALID表示不添加
conv = tf.nn.conv2d(
    input,filter_weight,strides=[1,1,1,1],padding="SAME"
)

#tf.nn.bias_add 提供了一个方便的函数给每一个节点加上偏置项，注意这里不能直接使用加法，因为矩阵上不同位置上的节点都需要加上
#同样的偏置项。
bias = tf.nn.bias_add(conv,biases)

#将结算结果通过Relu激活函数完成去线性化

actived_conv = tf.nn.relu(bias)
#一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
pool = tf.nn.max_pool(
    actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME"
)
