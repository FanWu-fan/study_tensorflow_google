import tensorflow as tf 

#使用tf.train.match_filenames_once函数来获取文件列表
files = tf.train.match_filenames_once("./path/to/data.tfrecords-*")

#通过tf.train.string_input_poduce函数创建输入队列，输入队列中的文件列表为
#tf.train.match_filenames_once 函数获取的文件列表。这里将shuffle参数设为False
#来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将 shuffle参数设置为 true
filename_queue = tf.train.string_input_producer(files,shuffle=False,num_epochs=1)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features =
    {
        'i':tf.FixedLenFeature([],tf.int64),
        'j':tf.FixedLenFeature([],tf.int64)
    }
)

with tf.Session() as sess:
    #虽然在本程序段中没有声明任何变量，但是用 tf.trian.match_filenames_once函数时需要
    #初始化一些变量
    tf.local_variables_initializer().run()
    print(sess.run(files))
    '''
[b'.\\path\\to\\data.tfrecords-00000-of-00002'
 b'.\\path\\to\\data.tfrecords-00001-of-00002']
    '''

    #声明tf.train.Corrdinator类来协同不同线程，并且启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'],features['j']]))
        '''
[0, 0]
[0, 1]
[1, 0]
[1, 1]
[0, 0]
[0, 1]
        '''
    coord.request_stop()
    coord.join(threads)