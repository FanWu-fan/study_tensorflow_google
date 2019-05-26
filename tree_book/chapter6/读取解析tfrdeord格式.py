import tensorflow as tf 
#解析一个 TFredord的方法。record是从文件中读取的一个样例。7.1节中具体介绍了如何解析TFrecord样例
#如何解析 TFRecord样例
def parser(record):
    #解析读入的一个样例
    features = tf.parse_single_example(
        record,
        features = {
            'i': tf.FixedLenFeature([],tf.int64),
            'j': tf.FixedLenFeature([],tf.int64)
        }
    )
    return features['i'],features['j']

#从 TFRecord 文件创建数据集
# files = tf.train.match_filenames_once("G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\\")
dataset = tf.data.TFRecordDataset(
    ["G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\data.tfrecords-00000-of-00002",
    "G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\data.tfrecords-00001-of-00002"
        ])

#map()函数表示对数据集中的每一条数据进行调用相应方法。使用 TFRecordDataset读出的
#是二进制的数据，这里需要通过map() 来调用 parse() 对二进制数据进行解析。类似地，
#map()函数也可以用来完成其他的数据预处理工作
dataset = dataset.map(parser)

#定义遍历数据集的迭代器
iterator = dataset.make_one_shot_iterator()

#feat1，feat2是 parser()返回的 一维的 int64型张量，可以作为输入用于进一步的计算
feat1,feat2 = iterator.get_next()

with tf.Session() as sess:
    # tf.local_variables_initializer().run()
    for i in range(20):
        f1,f2 = sess.run([feat1,feat2])
        print(f1,f2)
