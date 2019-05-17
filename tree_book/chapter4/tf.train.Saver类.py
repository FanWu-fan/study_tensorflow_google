import tensorflow as tf 

#声明两个变量并计算它们的和
v1 = tf.get_variable(
    "v1_tf",[1],initializer=tf.constant_initializer(1.0))

v2 = tf.get_variable(
    "v2_tf",[1],initializer=tf.constant_initializer(2.0))

result = v1 + v2 

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess,"./test_Saver/model.ckpt")

###################################################

with tf.Session() as sess:
    saver.restore(sess,"./test_Saver/model.ckpt")
    print(sess.run(result))# 3


