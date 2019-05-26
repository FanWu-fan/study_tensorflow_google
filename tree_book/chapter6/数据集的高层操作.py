import tensorflow as tf 

#列举输入文件，训练和测试使用不同的数据
train_files = tf.train.match_filenames_once("/path/to/train_file-*")
test_files = tf.train.match_filenames_once("/path/to/test_file-*")

#定义parser方法从TFRecord中解析数据。这里假设 image中存储的是图像的原始数据
# label 为该样例所对应的标签，height,width和channels给出了图片的维度
def parser(record):
    features = tf.parse_single_example(
        record,
        featires = {
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'height':tf.FixedLenFeature([],tf.int64),
            'width':tf.FixedLenFeature([],tf.int64),
            'channels':tf.FixedLenFeature([],tf.int64),
        }
    )

    #从原始图像数据解析出象素矩阵，并根据图像尺寸还原图像
    decode_image = tf.decode_raw(features['image'],tf.uint8)
    decode_image.set_shape([features['height'],features['width'],features['channels']])
    label = features['label']
    return decode_image,label

image_size = 299 #定义神经网络输入层图片的大小
batch_size = 100 #定义组合数据batch的大小
shuffle_buffer = 10000 #定义随机打乱数据时 buffer的大小

#定义读取训练数据的数据集
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

#对数据依次进行预处理，shuffle，和 batching操作。preprocess_for_train 为
#7.2.2小节中介绍的图像预处理程序。因为上一个mao得到的数据集中提供了 decoded_image
#和 label两个结果，所以这个map需要提供一个有2个参数的函数来处理数据，在下面的代码中，
#lambda中的image代表的就是第一个 map返回的 decoded_image,label代表的就是第一个map返回的label.
#在这个lambda表达式中我们首先将decoded_image在传入 proprocess_for_train来进一步对图像数据
#进行预处理。然后再将处理好的图像和label组成最终的输出

dataset = dataset.map(lambda image,label:(
    preprocess_for_train(image,image_szie,image_size,None),label
))
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

#重复Num_EPOCHS个 epoch,再上节 TRAINING_ROUNDS指定了训练的轮数，而在这里指定了
#整个数据集重复的次数，它也间接的确定了训练的轮数
NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

#定义数据集迭代器，虽然定义数据集时没有直接使用placeholder来提供文件地址，但是
#tf.trian.match_filenames_once 方法得到的结果和placeholder的机制类似
#也需要初始化，所以这里使用的是 initializable_iterator
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()

#定义神经网络结构以及优化过程
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#定义测试用的 Dataset,与训练不同，测试数据的Dataset不需要经过随机翻转等预处理操作，也不需要打乱顺序和
#重复多个 epoch，这里使用与训练数据先用的 parser进行解析，调整分辨率到网络输入层大小，然后直接进行 batching操作
test_dataset - tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(lambda image,label:
tf.image.resize_images(image,[image_size,image_size]),label))
test_dataset = test_dataset.batch(batch_size)

#定义测试数据上的迭代器
test_iterator = test_dataset.make_initializable_iterator()
test_image_batch,test_label_batch = test_iterator.get_next()

#定义预测结果为logit值最大的分类
test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit,axis=-1,output_type = tf.int32)

#声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer(),
    tf.local_variables_initializer())

    #初始化训练数据的迭代器
    sess.run(iterator.initializer)

    #循环进行训练，直到数据集完成输入、抛出 outOfRangeError错误
    while True:
        try:
            sess.run(train_step)
        except: tf.errors.OutOfRangeError:
            break
    
    #初始化测试数据的迭代器
    sess.run(test_iterator.initializer)

    #获取预测结果
    test_results = []
    test_labels = []
    while True:
        try:
            pred,label = sess.run([predictions,test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

#计算准确率
correct = [float (y==y_) for (y,y_) in zip (test_results,test_labels)
accuracy = sum(correct) / len(correct)

