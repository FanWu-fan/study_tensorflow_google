import tensorflow as tf
# 这里声明的变量名称和已经保存的变量的名称不同
v1 = tf.get_variable(
    "other-v1", [1], initializer=tf.constant_initializer(1.0))
v2 = tf.get_variable(
    "other-v2", [1], initializer=tf.constant_initializer(2.0)
)

result = v1+v2
# 如果直接使用tf.train.Saver()来加载模型会报找不到的错误
# Key other-v1 not found in checkpoint

save = tf.train.Saver({"v1_tf": v1, "v2_tf": v2})
with tf.Session() as sess:
    save.restore(sess, "./test_Saver/model.ckpt")
    print(sess.run(result))
