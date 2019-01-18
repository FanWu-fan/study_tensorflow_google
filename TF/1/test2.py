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
        with g1.device('/gpu:0'):
            print(sess.run(tf.get_variable("v")))

with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    #这里的initialize是动词，执行初始化所有变量的操作
    with tf.variable_scope("",reuse = True):
        print(sess.run(tf.get_variable("v")))

