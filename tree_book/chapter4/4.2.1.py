import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
# 输入层的节点数，对于MNIST数据集，这个就等于图片的像素
INPUT_NODE = 784

# 输出层的节点数，这个等于类别的数目。在这个区分0~9这10个数字
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例
# 这个隐藏层有500个节点
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数，数字越小，训练过程越接近
# 随机梯度下降；数字越大时，训练越接近梯度下降

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 10000  # 训练轮数
MOVING_AVERAGE_DEVAY = 0.99  # 滑动平均衰减率

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里
# 定义了一个使用RELU激活函数的三层全连接神经网络(输入层，一层隐藏层，输出层)，
# 通过加入隐藏层实现了多层网络结构，
# 通过RELU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类
# 这样方便在测试时使用 滑动平均模型

'''
#采用tf.variable_scope进行改进
def inference(input_tensor,reuse=False):
    with tf.variable_scope("layer1",reuse=reuse):
        #根据传进来的reuse来判断是创建新变量还是使用已经创建
        #好的。在第一次构造网络时需要创建新的变量，以后每次调用
        #这个函数都直接使用reuse=True就不需要每次将变量传进来了
        weights = tf.get_variable(
            "weights",[INPUT_NODE,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases = tf.get_variable(
            "biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

        #类似的定义第二层神经网络的变量和前向传播结果
    with tf.variable_scope("layer2",reuse=reuse):
        weights = tf.get_variable(
            "weights",[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1,weights)+biases
        
        #返回最后的前向传播结果
        return layer2
    
x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y = inference(x)
'''
############################################################################
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1 )

        # 计算输出层的前向传播结果，因为在计算损失函数时会一并计算 softmax 函数
        # 所以这里不用加入激活函数，而且不加入 softmax 不会影响预测结果。因为在
        # 预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果
        # 的计算没有影响，于是在计算整个神经网络的前向传播时可以不加入最后的 softmax 层

        return (tf.matmul(layer1, weights2) + biases2)

    else:
        # 首先使用 avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return (
            tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        )
#############################################################################
# 模型训练的过程


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数 正态分布，但如果随机出来的值偏离平均值超过2个标准差，
    # 那么这个数将会被重新随机
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
    biases1 = tf.Variable(
        tf.constant(0.1, shape=[LAYER1_NODE])
    )

    # 生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为 None,
    # 所以函数不会使用参数的 滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为
    # 不可训练的变量(trainable = False).在使用TF训练神经网络时，
    # 一般会将代表训练轮数的变量指定为 不可训练的 参数
    global_step = tf.Variable(0, trainable=False)

    # 给定 滑动平均衰减率 和 训练轮数的变量，初始化滑动平均类，设定滑动平均衰减率，和
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DEVAY, global_step
    )

    # 在所以代表神经网络参数的变量上使用 滑动平均。其他辅助变量(比如 global_step)就不
    # 需要了，tf.trainable_variables 返回的就是图上集合
    # tf.GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定
    # trainable = False的参数
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )

    '''
    计算使用了滑动平均之后的前向传播结果，滑动平均不会改变变量本身的取值，
    而是会维护一个影子变量来记录其滑动平均值。所以当需要使用这个滑动平均值时，
    需要明确调用 average 函数
    '''
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2
    )

    '''
    计算交叉熵作为刻画预测值和真实值之间差距的函数，这里使用了TF中提供的 
    sparse_softmax_cross_entropy_with_logits 函数来计算交叉熵，当分类问题
    只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中
    只包含了0~9中的一个数字，所以可以使用这个函数来计算交叉熵损失。这个函数的
    第一个参数是 神经网络不包括 softmax层的前向传播结果，第二个是训练数据的正确答案。
    因为标准答案是一个长度为10的一维数组，而该函数需要提供的是一个正确的答案数字，所以
    需要使用 tf.argmax 函数来得到正确答案对应的类别编号(返回最大值的那个数值的下标)。
    通过 np.argmax 实现 tf.argmax(y_,1) 和 tf.argmax(y_,0) 区别：
    '''
    #test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    # np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
   # np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
   # axis=0:是最大的范围，所有的数组都要进行比较，只是比较的是这些数组相同位置上的数：
   # axis=1:等于1的时候，比较范围缩小了，只会比较每个数组内的数的大小，结果也会根据有几个数组，产生几个结果。

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不是用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总的损失等于交叉熵损失和正则化 损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的
        # 学习率在这基础上递减
        global_step,  # 当前的迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率衰减速度
    )

    # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数，注意这里损失函数
    # 包含了交叉熵损失和 L2正则化损失
    # global_step=global_step 这样写以后 通过 minize可以实现 global_step自动加1
    # global_step: Optional `Variable` to increment by one after the
    # variables have been updated
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新 神经网络中的参数，
    # 又要更新每一个参数的 滑动平均值，为了一次完成多个操作，TF提供了两种机制
    # tf.control_dependencies  和  tf.group  下面两行程序等价
    # train_op = tf.group(train_step,variables_averages_op) 将多个操作合并成一个操作

    # def control_dependencies(control_inputs):control_dependencies是用于控制计算流图的先后顺序的。
    # 必需先完成control_input的计算，才能执行之后定义的context。
    # 但是，tensorflow是顺序执行的，为什么还需control_dependecies呢?
    # 原因在实际训练中，大多是以一个BATCH_SIZE大小来训练。因此需循环地去刷新前面所定义的变量。
    # 而采用control_dependencies可以确保control_input在被刷新之后,在执行定义的内容，从而保证计算顺序的正确性。

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y,1)
    # 计算每一个样例的预测答案，其中 average_y 是一个 batch_size*10的二维数组，每一行
    # 表示一个样例的前向传播结果，tf.argmax的第二个参数“1”表示选取最大值的操作仅在第一个
    # 维度中进行，也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为
    # batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。
    # tf.equal判断两个张量的每一维是否相等，如果相等返回 True, 否者返回 False

    correct_prediction = tf.equal(
        tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 这个运算首先将一个布尔型的数值转化为实数型，然后计算平均值，这个平均值
    # 就是模型在这一组数据上的正确率
    #tf.cast将bool转为float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据大致判断停止的
        # 条件和判断训练的结果

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果，因为MNIST数据集比较下，所以
                # 一次放入内存中
                validate_acc = sess.run(
                    accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy，using average model is %g" % (
                    i, validate_acc))

            # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run([train_op, global_step], feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        
        print("After %d training step(s), test accuracy using average,model is %g" % (
            TRAINING_STEPS, test_acc))

# 主程序入口


def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据集
    mnist = input_data.read_data_sets('path/to/MNIST_data', one_hot=True)
    train(mnist)


# TF提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()
