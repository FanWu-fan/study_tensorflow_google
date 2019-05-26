import tensorflow as tf 
#定于一个 LSTM结构，在TF中通过一句简单的命令就可以实现一个完整LSTM结构
#LSTM中使用的变量也会在该函数中自动被声明
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将LSTM中的状态初始化为全0数组，BasicLSTMCell类提供了zero_state函数来生成
#全0的初始状态，state是一个包含两个张量的LSTMStateTuple类，其中 state.c 和
#state.h 分别对应了c状态和h状态
#和其他神经网络类似，在优化循环神经网络时，每次也会使用一个 batch的训练脚本
#以下代码中，batch_size给出了一个 batch的大小
state  = lstm.zero_state(batch_size,tf.float32)

#定义损失函数
loss= 0.0

#虽然在测试时循环神经网络可以处任意长度的序列，但是在训练中为了将循环网络展开成 前馈神经网络，
#我们需要知道训练数据的序列长度。