import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

#使用rb,按照字节返回，由于不是tensor,因此不需要sess
image_raw_data = tf.gfile.FastGFile("./Images/Abyssinian_9.jpg",'rb').read()
# print(image_raw_data)

#返回的是 tensor,需要sess,这两个方法都是读取 jpeg的原始数据，需要进行解码
# image_data = tf.read_file("./Images/Abyssinian_9.jpg")
# print(image_data)#Tensor("ReadFile:0", shape=(), dtype=string)

# with tf.Session() as sess:
#     print(sess.run(image_data))

with tf.Session() as sess: 
    image_data  = tf.image.decode_jpeg(image_raw_data,channels=1)
    # image_data = sess.run(image_data)
    print(image_data)
    print(image_data.shape)
    h,w,c = image_data.shape
    assert c ==1
    image_data = image_data.reshape(h,w)
    plt.imshow(image_data,cmap='gray')
    plt.show()

