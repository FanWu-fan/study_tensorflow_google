import tensorflow as tf 

#定义一个变量用于计算滑动平均，这个变量的初值为0，注意这里手动指定了变量的
#类型为tf.float32,因为所有需要计算滑动平均的变量必须是 实数型
v1 = tf.Variable(0,dtype=tf.float32)

#这里step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率,step即为num_updates参数
step = tf.Variable(0,trainable=False)

#定义一个滑动平均类(class)。初始化时给定了衰减率(0.99)和控制衰减率
#的变量step
ema = tf.train.ExponentialMovingAverage(0.99,step)

#定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
#这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有的变量
    tf.global_variables_initializer().run()

    #获取ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1和v1的
    #滑动平均都为0
    print(sess.run([v1,ema.average(v1)])) #输出[0.0.0.0]

    #更新变量v1的值到5
    sess.run(tf.assign(v1,5))

    #更新v1的滑动平均值。衰减率为 min{0.99, (1+step)/(10+step)=0.1}=0.1
    #所以v1的滑动平均会被更新为 0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) #输出为[5.0,4.5]

    #更新step的值为10000
    sess.run(tf.assign(step,10000))
    #更新v1的值为10
    sess.run(tf.assign(v1,10))
    #更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step)=0.999} = .099
    #所以v1的滑动品均会被跟新为 0.99*4.5+0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) #输出为[10,4.5549998]

    #再次更新滑动平均值，得到的新滑动平均值为 0.99*4.55+0.01*10=4.60
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) #输出为[10,4.60]

    
