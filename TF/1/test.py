import tensorflow as tf 

def A():
    a = tf.constant([1.0, 2.0], name = "a")
    b = tf.constant([2.0, 3.0], name = "b")
    #result = a + b
    result = tf.add(a, b, name = "add")
    # print(result)
    #print(tf.Session().run(result))
    # tf.Session().close()

    # with sess.as_default():
    #     print(result.eval())

    # print(sess.run(result))
    # print(result.eval(session = sess))
    # tf.Session().close()

    # sess = tf.InteractiveSession()
    # print(result.eval())
    # sess.close()
    
    config = tf.ConfigProto(allow_soft_placement = False,
                            log_device_placement = True)
    sess1 = tf.InteractiveSession(config = config)
    sess2 = tf.Session(config = config)
    print(result.eval())

def B():
    #声明w1,w2两个变量，这里还通过seed参数设定了随机种子
    #这样可以保证每次运行得到的结果是一样的
    w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed =1))
    w2 = tf.Variable(tf.random_normal([3,1], stddev =1, seed =1))

    #暂时将输入的特征向量定义为一个常量，注意这里x是一个1*2的矩阵
    x = tf.constant([[0.7,0.9]])

    #通过矩阵乘法
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    # sess = tf.Session()
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    # print(sess.run(y))
    # sess.close()

   
    # with tf.Session() as sess:
    #     sess.run(w1.initializer)
    #     sess.run(w2.initializer)
    #     print(y.eval())

    # with tf.Session() as sess:
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #     print(sess.run(y))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(y))

if __name__ == "__main__":
    B()