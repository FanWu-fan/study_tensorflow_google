import tensorflow as tf 
from numpy.random import RandomState
#这里通过Numpy工具包生成模拟数据集

batch_szie = 8

#两个输入节点
x = tf.placeholder(
    tf.float32, shape = (None,2), name = 'x-input')
y_ = tf.placeholder(
    tf.float32, shape=(None,1),name = 'y-input'
)

#定义了一个单层的神经网络前向传播的过程，就是简单的加权和。
#shape: 输出张量的形状，必选 mean: 正态分布的均值，默认为0
#stddev: 正态分布的标准差，默认为1.0 
#seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
#dtype: 输出的类型，默认为tf.float32
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义预测多了和少了的成本。
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(
    tf.where(tf.greater(
        y,y_),(y-y_)*loss_more,(y_-y)* loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm  = RandomState(1)#产生一个随机状态种子
data_size = 128
X = rdm.rand(data_size, 2)#生成 128行2列的数据
#设置回归的正确值为两个输入的和加上一个随机量，之所以要加上
#一个随机量是为了加入不可预测的噪音，否者不同的损失函数的意义就
#不大，因为不同的损失函数都会在能完全预测正确的时候最低。一般
#来说噪音为一个均值为0的小量，所以这里的噪音设置为
#-0.05~0.05的随机数
Y= [[x1 + x2 + rdm.rand() / 10.0-0.05] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    init_op.run()
    
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_szie) % data_size
        end = min(start+batch_szie,data_size)
        
        sess.run(
            train_step,feed_dict={x:X[start:end],y_:Y[start:end]}
        )
    print(sess.run(w1))
