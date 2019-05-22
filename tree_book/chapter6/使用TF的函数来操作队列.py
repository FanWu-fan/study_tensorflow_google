import tensorflow as tf 

# #创建一个先进先出的队列，指定队列中最多可以保存两个元素，并指定
# #类型为整数
# q = tf.FIFOQueue(capacity=2,dtypes=tf.int32)

# #使用enqueue_many 函数初始化队列中的元素，和变量初始化类似，在使用队列之前
# #需要明确的调用这个初始化过程
# init = q.enqueue_many(([0,10],))

# #使用 Dequeue 函数将队列中的第一个元素出队列，这个函数的值将被存在变量x中。
# x = q.dequeue()
# #将得到的值加1
# y = x+1
# #将加1 的值再重新加入队列中
# q_inc = q.enqueue([y])

# with tf.Session() as sess:
#     #运行处死花花队列的操作
#     init.run()
#     for _ in range(7):
#         # 运行q_inc将执行数据出队列，出队的元素+1，重新加入队列的整个过程。
#         v,_=sess.run([x,q_inc])
#         #打印出队元素的取值
#         print(v) #0  10  1  11  2

##############################################################
'''
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32])
# init = q.enqueue_many(input_data) #3.0 2.0 1.0
init = q.enqueue(input_data)
out_data = q.dequeue()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init)
    sess.run(init)

    print("1: ",sess.run(out_data))
    print("2: ",sess.run(out_data))
    print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))
# 1:  [3. 2. 1.]
# 2:  [3. 2. 1.]
# 3:  [3. 2. 1.]
# True
'''
###############################################################
'''
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
input_data1 = [ [33.,22.,11.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32])
init = q.enqueue(input_data)
init1 = q.enqueue(input_data1)

output_data = q.dequeue()
with tf.Session() as sess:
    init.run()
    init1.run()
    init1.run()

    print('1：',sess.run(output_data))
    print('2：',sess.run(output_data))
    print('3：',sess.run(output_data))
    sess.run(q.close(cancel_pending_enqueues=True))
    print(sess.run(q.is_closed()))
# 1： [3. 2. 1.]
# 2： [33. 22. 11.]
# 3： [33. 22. 11.]
# True
'''
#####################################################
'''
# enqueue()每次入列一个元素，对于同一个输入数据，多次入列会重复入列相同的元素
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
# q = tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32],shapes=[[],[]])
# 1:  [3.0, 11.0]
# 2:  [2.0, 22.0]
# 3:  [1.0, 33.0]
# True

q = tf.FIFOQueue(3,dtypes = [tf.float32],shapes = [[]])
# 1:  3.0
# 2:  2.0
# 3:  1.0
# True

init = q.enqueue_many(input_data) #3.0 2.0 1.0
# init = q.enqueue(input_data)
out_data = q.dequeue()

with tf.Session() as sess:
    sess.run(init)
    # sess.run(init)
    # sess.run(init)

    print("1: ",sess.run(out_data))
    print("2: ",sess.run(out_data))
    print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))
'''
#################################################################
'''
# enqueue()每次入列一个元素，对于同一个输入数据，多次入列会重复入列相同的元素
input_data = [ [3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
q = tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32],shapes=[[],[]])
init = q.enqueue_many(input_data) 

out_data = q.dequeue_many(2)

with tf.Session() as sess:
    sess.run(init)
    # sess.run(init)
    # sess.run(init)

    print("1: ",sess.run(out_data))
    # print("2: ",sess.run(out_data))
    # print("3: ",sess.run(out_data))
    sess.run(q.close(cancel_pending_enqueues=False))
    print(sess.run(q.is_closed()))

# 如上例队列的元素为 [3.0, 11.0]，[2.0, 22.0]，[1.0, 33.0]，
# dequeue_many(2)将队列元素的第0维3.，2.，1.中取2个数据组成[3., 2.]，然后11.，22.，33.变为第0维，
# 再取2个数据组成[11., 22.]，所以出列的数据为[[3., 2.],[11., 22.],]
'''
#####################################################################

import tensorflow as tf

input_data=[[[3.,2.,1.],[11.,22.,33.],[111.,222.,333.]],[[23.,22.,21.],[211.,222.,233.],[2111.,2222.,2333.]]]
print(tf.shape(input_data))
q=tf.FIFOQueue(3,tf.float32)
init=q.enqueue_many(input_data)
output_data=q.dequeue()
with tf.Session() as sess:
    init.run()
    print('1：',sess.run(output_data))
    print('2：',sess.run(output_data))
    print('3：',sess.run(output_data))
    sess.run(q.close(cancel_pending_enqueues=True))
    print(sess.run(q.is_closed()))

'''
1： [3. 2. 1.]
2： [11. 22. 33.]
3： [111. 222. 333.]
True
输入数据input_data=[[[3.,2.,1.],[11.,22.,33.],[111.,222.,333.]],[[23.,22.,21.],[211.,222.,233.],[2111.,2222.,2333.]]]
为三维，所有第0维的张量为[3. 2. 1.]，[11. 22. 33.]， [111. 222. 333.]，enqueue_many会将他们组合在一起输入队列，
所以队列的元素为[3. 2. 1.]，[11. 22. 33.]， [111. 222. 333.]
'''


