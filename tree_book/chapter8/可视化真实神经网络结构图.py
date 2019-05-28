import tensorflow as tf 
import os
from tensorflow.examples.tutorials.mnist import input_data
#mnist_inference中定义的常量和前向传播的函数不需要改变，因为前向传播已经通过
#tf.variable_scope实现了计算节点按照网络结构的划分
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNINT_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = ".\model"
MODEL_NAME = "model.ckpt"

import mnist_inference

def train(mnist):
    #将处理输入数据的计算都放在名字为“input1”的命名空间下
    with tf.variable_scope("input"):
        x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input"
    )
        y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input"
    )
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        y = mnist_inference.inference(x, regularizer)
        global_step = tf.get_variable(
        "global_step",initializer=0, trainable=False)
    
    #将处理滑动平均相关的计算都放在名为 moving_average的命名空间下
    with tf.variable_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #将计算损失函数相关的计算都放在名为 loss_function的命名空间下
    with tf.variable_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
       logits = y, labels = tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #将定义学习率，优化方法，以及每一轮训练需要执行的操作都放在名字为“train_step”
    #的命名空间下
    with tf.variable_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNINT_RATE_DECAY
    )

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )
        with tf.control_dependencies([train_step,variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 初始化TF持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不在测试模型在验证数据上的表现，验证和测试的过程将会有一个
        # 独立的程序来完成
        # _,对应的是3个返回值
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            # _,loss_value,step = sess.run(
            #     [train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            if i%1000 == 0:
                #配置运行时需要记录的信息
                run_options = tf.RunOptions(
                    trace_level = tf.RunOptions.FULL_TRACE
                )
                #运行时需要记录信息的proto
                run_metadata = tf.RunMetadata()
                #将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点
                #的时间，空间开销信息
                _,loss_value,step = sess.run(
                    [train_op,loss,global_step],feed_dict = {x:xs,y_:ys},
                    options=run_options,run_metadata=run_metadata
                )
                #将节点在运行时的信息写入日志文件
                writer = tf.summary.FileWriter("./path/to/loga",tf.get_default_graph())
                writer.add_run_metadata(run_metadata,"step%03d"%i)
                # 输出当前的训练情况。这里只输出了模型在当前batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练的情况，在验证数据集上的正确率信息会有
                # 一个单独的程序来生成
                print("After %d training step(s),loss on training batch is %g."%(step,loss_value))
                # os.path.join路径拼接
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            else:
                _,loss_value,step = sess.run(
                [train_op,loss,global_step],feed_dict={x:xs,y_:ys})
    
    #将当前的计算图输出到 TB的日志文件
    writer.close()

def main(argv = None):
    mnist = input_data.read_data_sets('path/to/MNIST_data',one_hot = True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()
