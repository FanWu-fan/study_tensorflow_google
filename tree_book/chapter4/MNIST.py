from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("path/to/MNIST_data",one_hot = True)

# 打印Traning data size: 55000
print("Traning data size: ",mnist.train.num_examples)

# 打印Validating data size: 5000
print("Validating data size: ", mnist.validation.num_examples)

# 打印Testing data size: 10000
print("Testing data size: ",mnist.test.num_examples)

# 打印Example trainging data label:
print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)
#从train的集合中选取batch_size个训练数据

print("X shape: ",xs.shape)
#(100,784)

print("Y shape: ",ys.shape)
#(100,10)
