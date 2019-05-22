import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np

#读取原生图像的数据
image_raw_data = tf.gfile.FastGFile("./Images/Abyssinian_9.jpg",'rb').read()

with tf.Session() as sess:
    #将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。TF还提供了
    #tf.image.decode_png函数对png格式的图像进行解码，解码之后的结果为
    #张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)

    # print(img_data.eval())

    # #使用pyplot工具可视化得到的图像。
    # plt.imshow(img_data.eval())
    # plt.show()

    # # 将数据的类型转化为实数方便下面的样例程序对图像进行处理,如果不转化为实数类型的化，那么会
    #在下面运算得出为 [0-255]的浮点数，不能直接显示图片，因为 plt需要的参数是[0-255]的Int,以及[0-1]的float
    img_data = tf.image.convert_image_dtype(image=img_data,dtype=tf.float32)

    # #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中，打开这张图像，
    # #可以得到和原始图像一样的图像
    # encod_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile("./Images/9.jpg","wb") as f:
    #     f.write(encod_image.eval())

    # #通过tf.image.resize_images函数调整图像的大小，这个函数第一个参数为原始图像
    # #第二个和第三个参数为调整后的图像的大小，method参数给出了调整图像大小的方法
    resized = tf.image.resize_images(img_data,[300,300],method=0)

    # #输出调整后图像的大小，此处的结果为(300,300,?),图像深度在未明确设定之前是问号
    
    print(resized)
    # resized = np.asarray(resized.eval(),dtype = 'uint8')
    # resized = tf.image.convert_image_dtype(resized,dtype=tf.uint8)
    resized = sess.run(resized)
    print(resized)
    # plt.imshow(sess.run(resized))
    # plt.show()

    # #通过tf.image.resize_image_with_crop_or_pad 函数调整图像的大小，这个函数的
    # #第一个参数为原始图像，后面的两个参数是调整后的目标图像大小。如果原始图像的尺寸
    # #大于目标图像，那么这个函数会自动在原始图像的四周填充全0背景。
    # croped = tf.image.resize_image_with_crop_or_pad(img_data,300,300)
    # padded = tf.image.resize_image_with_crop_or_pad(img_data,700,700)
    # print(croped.get_shape())
    # print(padded.get_shape())
    # croped = np.asarray(croped.eval(),dtype='uint8')
    # padded = np.asarray(padded.eval(),dtype='uint8')

    # plt.imshow(croped)
    # plt.show()
    # plt.imshow(padded)
    # plt.show()

    # #图像通过比例调整图像大小,
    # central_cropped = tf.image.central_crop(img_data,0.5)
    # plt.imshow(central_cropped.eval())
    # plt.show()

    # #图像翻转
    # #上下翻转
    # flipped = tf.image.flip_up_down(img_data)
    # plt.imshow(flipped.eval())
    # plt.show()

    # #左右
    # filpped_LR = tf.image.flip_left_right(img_data)
    # plt.imshow(filpped_LR.eval())
    # plt.show()

    # #对角线
    # transposed = tf.image.transpose_image(img_data)
    # plt.imshow(transposed.eval())
    # plt.show()


    # #以一定的概率上下翻转图像
    # flipped = tf.image.random_flip_left_right(img_data)
    # #以一定概率左右反转图像
    # flipped_LR = tf.image.random_flip_left_right(img_data)

    # plt.imshow(flipped.eval())
    # plt.show()
    # plt.imshow(flipped_LR.eval())
    # plt.show()
    
    #调整图像的色彩
    # adjusted = tf.image.adjust_brightness(img_data,-0.5)
    # plt.imshow(adjusted.eval())
    # plt.show()

    # adjusted = tf.image.adjust_brightness(img_data,0.5)
    # plt.imshow(adjusted.eval())
    # plt.show()

    #在[-max_delta,max_delta]的范围内随机调整图像的亮度
    # adjusted = tf.image.random_brightness(img_data,max_delta=1)
    # plt.imshow(adjusted.eval())
    # plt.show()

    #调整图像的对比度
    # adjusted = tf.image.adjust_contrast(img_data,-5)
    # plt.imshow(adjusted.eval())
    # plt.show()