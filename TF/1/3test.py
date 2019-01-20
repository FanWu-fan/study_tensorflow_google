import tensorflow as tf

def A():
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1),dtype=tf.float64)
    w2 = tf.Variable(tf.random_normal([2,3],stddev=1,dtype=tf.float64))

    w1.assign(w2)








if __name__ == "__main__":
    A()