import tensorflow as tf 

#解析一个TFRecord的方法
def parser(record):
    features = tf.parse_single_example(
        record,
        features = {
            'i':tf.FixedLenFeature([],tf.int64),
            'j':tf.FixedLenFeature([],tf.int64),
        }
    )
    return features['i'],features['j']

files = tf.train.match_filenames_once("G:\Code\study_tensorflow_google\\tree_book\chapter6\path\\to\*")
input_files = tf.placeholder(tf.string) 
dataset = tf.data.TFRecordDataset(input_files)
dataset= dataset.map(parser)

#定义遍历dataset的initializeble_iterator
iterator = dataset.make_initializable_iterator()
feat1,feat2 = iterator.get_next()

with tf.Session() as sess:
    #首先初始化interator,并给出input_files的值
    tf.local_variables_initializer().run()
    print(files)
    sess.run(iterator.initializer,
    feed_dict = {
        input_files:files.eval
    })
    #遍历所有数据一个 epoch.当遍历结束时，程序会抛出OutofrangeError
    while True:
        try:
            f1,f2 = sess.run([feat1,feat2])
            print(f1,f2)
        except tf.errors.OutOfRangeError:
            break

