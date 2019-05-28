import tensorflow as tf 

a = tf.constant([1.,2.,3.],shape=[3],name='a')
b = tf.constant([1.,2.,3.],shape=[3],name='b')
c=a+b
#通过log_device_placement参数来输出运行每一个运算的设备
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
print(sess.run(c))

'''
在没有GPU上运行
add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
2019-05-28 16:04:35.023846: I tensorflow/core/common_runtime/placer.cc:927] add: (Add)/job:localhost/replica:0/task:0/device:CPU:0
a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2019-05-28 16:04:35.027409: I tensorflow/core/common_runtime/placer.cc:927] a: (Const)/job:localhost/replica:0/task:0/device:CPU:0
b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2019-05-28 16:04:35.030455: I tensorflow/core/common_runtime/placer.cc:927] b: (Const)/job:localhost/replica:0/task:0/device:CPU:0
[2. 4. 6.]
'''