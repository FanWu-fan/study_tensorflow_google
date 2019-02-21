import tensorflow as tf 
# def A():
#     a = tf.constant([1.0, 2.0], name = "a")
#     b = tf.constant([2.0, 3.0], name = "b")
#     #result = a + b
#     result = tf.add(a, b, name = "add")
#     print(result)
# A()
# #输出：Tensor("add:0", shape=(2,), dtype=float32)
# #result这个张量是计算节点“add"输出的第一个结果
# #是一个一维数组，数组的长度为2
##########################################################

'''
#声明w1,w2两个变量，这里还通过seed参数设定了随机种子
#这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed =1))
w2 = tf.Variable(tf.random_normal([3,1], stddev =1, seed =1))

 #暂时将输入的特征向量定义为一个常量，注意这里x是一个1*2的矩阵
x = tf.constant([[0.7,0.9]])

 #通过矩阵乘法
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# with tf.Session() as sess:
#     sess.run(w1.initializer)
#     sess.run(w2.initializer)
#     print(y.eval())

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(y))
'''
###################################################
'''
w1 = tf.Variable(tf.random_normal([2,3], stddev=1,seed = 1), name = 'w1')
w2 = tf.Variable(tf.random_normal([2,3], stddev=1,dtype= tf.float64, seed =1),
                name = 'w2')
# w1.assign(w2)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('w2: ', sess.run(w2))
    print('w1: ', sess.run(w1))
''' 
###################################################
'''
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1),
            name = 'w1')
w2 = tf.Variable(tf.random_normal([2, 2], stddev= 1),
            name = 'w2')
#下面这句话会报维度不匹配的错误：
#ValueError：
#tf.assign(w1,w2)
#下面这句话可以执行成功
tf.assign(w1, w2, validate_shape=False)
'''
###################################################
'''
w1 = tf.Variable(tf.random_normal([2, 3], stddev =1))
w2 = tf.Variable(tf.random_normal([3, 1],stddev = 1))

#定义 placeholder 作为存放数据的地方，这里维度不一定要定义
# 但是如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(1,2), name = "input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #print(sess.run(y)) #这个会报错InvalidArgumentError (see above for traceback): 
    #You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]

    print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))
    '''
###################################################
'''
w1 = tf.Variable(tf.random_normal([2, 3], stddev =1))
w2 = tf.Variable(tf.random_normal([3, 1],stddev = 1))
x = tf.placeholder(tf.float32)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #print(sess.run(y)) #这个会报错InvalidArgumentError (see above for traceback): 
    #You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]

    print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
'''
###################################################
cross_entropy = -tf.reduce_mean(
    y_* tf.log(tf.clip_by_value(y,1e-10,1.0))
)
learning_rate = 0.001
train_step = \
    tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
###################################################