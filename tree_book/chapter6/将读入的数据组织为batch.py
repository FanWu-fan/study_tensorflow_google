import tensorflow as tf 

#使用tf.train.match_filenames_once函数来获取文件列表
files = tf.train.match_filenames_once("./path/to/data.tfrecords-*")

#通过tf.train.string_input_poduce函数创建输入队列，输入队列中的文件列表为
#tf.train.match_filenames_once 函数获取的文件列表。这里将shuffle参数设为False
#来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将 shuffle参数设置为 true
filename_queue = tf.train.string_input_producer(files,shuffle=False)

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
   

    example,label = features['i'],features['j']

    #一个batch中样例的个数
    batch_size =3
    #组合样例的队列中最多可以存储的样例个数，这个队列如果太大，那么需要占用许多内存资源：
    #如果太小，那么出队操作可能会因为没有数据而被阻碍(block),从而导致训练效率降低。一般来说
    #这个队列的大小会和每一个batch的大小有关，下面一行代码给出了设置队列大小的一种方式
    capacity = 1000 + 3*batch_size

    #使用tf.train.batch函数来组合样例，[example,label]参数给出了需要组合的元素，一般example和
    #label分别代表 训练样本和这个样本对应的正确标签。batch_size参数给出了每个batch中样例的个数。
    # capacity给出了队列最大的容量，当队列长度等于容量时，TF将暂停入队操作，而只是等待元素出队。
    #元素个数小于容量时，TF重新启动入队
    example_batch,label_batch = tf.train.batch(
        [example,label],batch_size=batch_size,capacity=capacity
    )

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        #获取打印组合之后的样例。在真实问题中，这个输出一般会作为神经网络的输入
        for i in range(2):
            cur_example_batch,cur_label_batch = sess.run(
                [example_batch,label_batch]
            )
            print(cur_example_batch,cur_label_batch)
        coord.request_stop()
        coord.join(threads)       