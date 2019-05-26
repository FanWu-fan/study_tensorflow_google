import tensorflow as tf 

#从一个数组创建数据集
input_data = [1,2,3,4,5,6,7,8]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

#定义一个迭代器用于遍历数据集。因为上面定义的数据集没有用 palceholder
#作为输入参数，所以这里可以使用最简单的 one_shot_iterator
iterator = dataset.make_one_shot_iterator()

# get_next()返回表示一个输入数据的张量，类似于队列的 dequeue()
x = iterator.get_next()
y = x*x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
'''
1
4
9
16
25
36
49
64
'''
