import tensorflow as tf 

def A():
    a = tf.constant([1.0, 2.0], name = "a")
    b = tf.constant([2.0, 3.0], name = "b")
    #result = a + b
    result = tf.add(a, b, name = "add")
    # print(result)
    #print(tf.Session().run(result))
    # tf.Session().close()

    # with sess.as_default():
    #     print(result.eval())

    # print(sess.run(result))
    # print(result.eval(session = sess))
    # tf.Session().close()

    # sess = tf.InteractiveSession()
    # print(result.eval())
    # sess.close()
    
    config = tf.ConfigProto(allow_soft_placement = False,
                            log_device_placement = True)
    sess1 = tf.InteractiveSession(config = config)
    sess2 = tf.Session(config = config)
    print(result.eval())



def B():
    pass




if __name__ == "__main__":
    A()