import tensorflow as tf 

#创建 TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


#模拟海量数据情况下将数据写入不同的文件。num_shards 定义了总共写入多少个文件
# instances_per_shard 定义了每个文件中有多少个数据
num_shards = 2
isinstances_per_shard =2

for i in range(num_shards):
    #将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。其中m
    #表示了数据总共被存在了多少个文件中，n表示了当前文件的编号。式样的方式既方便了
    #通过正则表达式获取文件列表，又在文件中加入了更多的信息。
    filename = ('./path/to/data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    #将数据封装成EXample结构并写入TFRECOed格式
    for j in range(isinstances_per_shard):
        #Example结构仅包含当前样例属兔第几个文件以及当前文件的第几个样本
        example = tf.train.Example(features= tf.train.Features
        (feature = {
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()



#####################################################################################################
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# print(dataset1.output_types)  # ==> "tf.float32"
# print(dataset1.output_shapes)  # ==> "(10,)"



# dataset2 = tf.data.Dataset.from_tensor_slices(
#    (tf.random_uniform([4]),
#     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
# print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
# print(dataset2.output_shapes)  # ==> "((), (100,))"

# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
# print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"


#############################################################################################################
