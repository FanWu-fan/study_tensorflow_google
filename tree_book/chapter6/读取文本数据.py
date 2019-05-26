import tensorflow as tf 

#从文本文件创建数据集。假定每行文字时一个训练例子，注意这里可以提供多个文件。
files = tf.train.match_filenames_once('G:\许的数据集\csv1\\gc*')
dataset = tf.data.TextLineDataset(files)

# dataset = tf.data.TextLineDataset(['G:\许的数据集\csv1\\gc1.csv','G:\许的数据集\csv1\\gc2.csv'])


#定义迭代器用于遍历数据集
# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator(tf.string)

#这里get_next()返回一个字符串类型的张量，代表文件中的一行

x = iterator.get_next()
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    
    # print(sess.run(files))
    for i in range(3):
        print(sess.run(x))