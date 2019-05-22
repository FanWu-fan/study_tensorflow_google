'''
import tensorflow as tf 
import numpy as np 
import threading
import time

#线程中运行的程序，这个程序每隔1秒判断是否需要停止并打印自己的ID
def Loop(coord,worker_id):
    #使用tf.Coordinator 类提供的协同工具判断线程是否需要停止
    while not coord.should_stop():
        #随机停止所有的线程
        if np.random.rand() < 0.1:
            print("stoping id: %d"%worker_id)
            #调用coord.request_stop()函数来通知其他线程停止
            coord.request_stop()
        else:
            #打印当前id
            print("working on id: %d"%worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()

threads = [threading.Thread(target=Loop,args=(coord,i)) for i in range(5)]

for t in threads:t.start()
coord.join(threads)
'''
###########################################################################################
import tensorflow as tf 

#声明一个先进先出的队列，最多100个元素
queue = tf.FIFOQueue(100,"float")
#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#使用 tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner的第一个参数给出了倍操作的列表 [enqueue_op]*5
#表示了需要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

#将定义过的 QueueRunner加入TF计算图上指定的集合
# tf.train.add_queue_runner函数没有指定集合，
# 则加入默认集合 tf.GraphKeys.QUEUE_RUNNERS.下面的函数就是刚刚定义的qr加入
tf.train.add_queue_runner(qr)

#定义出队操作
out_tensor= queue.dequeue()

with tf.Session() as sess:
    # 使用 tf.train.Coordinator来协同启动的进程
    coord = tf.train.Coordinator()
    #使用 tf.train.QueueRunner时，需要明确调用 tf.train.start_queue_runners
    #来启动所有线程，否者因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队
    #操作被执行。tf.train.start_queue_runners函数会默认启动 tf.GraphKeys.QUEUE_RUNNERS集合中
    #的所有 QueueRunner. 因为这个函数支支持启动集合中的 QueueRunner，所以一般来说 tf.train.add_queue_runner
    # 函数和 tf.train.start_queue_runners 函数会指向同一个集合
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    #获取队列中的取值
    for _ in range(3):print(sess.run(out_tensor)[0])

    #使用tf.train.Coordinator来停止所有的线程
    coord.request_stop()
    coord.join(threads)
    '''
    上面的程序将启动五个线程来执行队列入队的操作，每个线程都是将随机数写入队列，
0.7624912
-0.8749699
-1.698716
'''