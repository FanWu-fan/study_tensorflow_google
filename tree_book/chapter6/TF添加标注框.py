import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("./Images/Abyssinian_9.jpg",'rb').read()



with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    #tf.image.draw_bouding_boxes函数要求图像矩阵中的数字为实数，所以需先将图像矩阵转化为
    #实数类型。tf.image.draw_bouding_boxes函数图像的输入是一个 batch的数据，也就是多张图像
    #组成的 四维矩阵，所以需要将解码后的图像矩阵加一维。
    # batched = tf.expand_dims(
    #     tf.image.convert_image_dtype(img_data,tf.float32),0
    # )

    #给出每一张图像的所有标注框，一个标注框有四个数字，分别代表[ymin,xmin,ymax,xmax]
    #注意这里给出的数字都是图像的相对位置，比如在 180*267的图像中
    #[0.35,0.47,0.5,0.56]-->[63,125]到[90,150]的图像
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    # result = tf.image.draw_bounding_boxes(batched,boxes)
    # result = tf.image.convert_image_dtype(result,tf.uint8)
    # result = sess.run(result)
    # bat,h,w,c = result.shape
    # result = result.reshape(h,w,3)
    # plt.imshow(result)
    # plt.show()
    begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data),bounding_boxes=boxes
    )
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data,tf.float32),0
    )

    result = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
    result = tf.image.convert_image_dtype(result,tf.uint8)
    result = sess.run(result)
    bat,h,w,c = result.shape
    result = result.reshape(h,w,3)
    plt.imshow(result)
    plt.show()



    #截取随机出来的图像。
    distorted_image = tf.slice(img_data,begin,size)
    print(distorted_image)
    distorted_image = sess.run(distorted_image)
    print(distorted_image)
    plt.imshow(distorted_image)
    plt.show()