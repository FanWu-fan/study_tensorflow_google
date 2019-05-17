import tensorflow as tf 
saver = tf.train.import_meta_graph(
    "./test_Saver/model.ckpt.meta"
)

with tf.Session() as sess:
    saver.restore(sess,"./test_Saver/model.ckpt")
    #通过张量名称获取张量
    print(sess.run(
        tf.get_default_graph().get_tensor_by_name("add:0")))
        #输出3