import time
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py和mnist_train.py中定义的常量和函数
import mnist_inference
import mnist_train

#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS =10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32,[None,mnist_inference.INPUT_NODE],name = 'x-input'
        )
        y_ = tf.placeholder(
            tf.float32,[None,mnist_inference.OUTPUT_NODE],name = 'y-input'
        )
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        #直接通过调用封装好的函数计算前向传播的结果，因为测试时不关心正则化损失的值
        #所以这里用于计算正则化损失的函数被设置为 None
        y = mnist_inference.inference(x,None)

        correction_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

        #通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用 滑动平均函数来
        #获取平均值了，这样就可以完全共用mnsit_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        #{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
        varialbles_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(varialbles_to_restore)

        #每个EVAL_INTERVAL_SECS 秒调用一次计算正确率的过程以检查训练过程中正确率的
        #变化
        while True:
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state函数会通过checkpoint文件自动
                #找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict = validate_feed)
                    print("After %s training step(s),validation accuracy = %g"%(global_step,accuracy_score))

                else:
                    print('No check point file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)
                
def main(argv = None):
    mnist = input_data.read_data_sets('path/to/MNIST_data',one_hot = True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()