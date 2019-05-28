import tensorflow as tf 

#通过 tf.device将运算指定到特定的设备上
with tf.device('cpu/:0'):
    a = tf.constanta = tf.constant([1.,2.,3.],shape=[3],name='a')
    b = tf.constant([1.,2.,3.],shape=[3],name='b')

with tf.device('/gpu:1'):
    c=a+b

sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
print(sess.run(c))

'''
add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
2019-05-28 16:10:47.670763: I tensorflow/core/common_runtime/placer.cc:1059] add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2019-05-28 16:10:47.670807: I tensorflow/core/common_runtime/placer.cc:1059] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2019-05-28 16:10:47.670826: I tensorflow/core/common_runtime/placer.cc:1059] b: (Const)/job:localhost/replica:0/task:0/device:CPU:0
[2. 4. 6.]
'''