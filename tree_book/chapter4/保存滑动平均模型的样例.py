import tensorflow as tf 

v = tf.get_variable("v",dtype=tf.float32,initializer=0.0)
#再没有申明滑动平均模型时只有一个变量v，所以下面的语句只会
#输出“v:0”

for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(decay = 0.99)

maintain_averages_op = ema.apply(tf.global_variables())

#在声明话哦的那个平均模型之后，TF会自动生成一个 影子变量
#输出：v:0
# v/ExponentialMovingAverage:0
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    #保存时，TF会将v:0 v/ExponentialMovingAverage:0两个变量都存下来
    saver.save(sess,"./ExponentialMovingAverage_Saver/model.ckpt")
    print(sess.run([v,ema.average(v)]))#[10.0, 0.099999905]

#############################################################################
#v = tf.get_variable("v",dtype=tf.float32,initializer=0.0)
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"./ExponentialMovingAverage_Saver/model.ckpt")
    print(sess.run(v))#0.099999905
############################################################################
print(ema.variables_to_restore())
#{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"./ExponentialMovingAverage_Saver/model.ckpt")
    print(sess.run(v))#0.099999905




