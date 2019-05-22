import tensorflow as tf 

x = tf.constant([[[1.,1.],[2.,2.],[3.,3.]]])
print(x.get_shape())#(1,3,2)
y = tf.constant([[1.,1.],[2.,2.]])
print(y.get_shape())#(2,2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(tf.reduce_mean(y,axis=1).eval())