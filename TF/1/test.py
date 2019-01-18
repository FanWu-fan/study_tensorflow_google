import tensorflow as tf 

def A():
    a = tf.constant([1.0, 2.0], name = "a")
    b = tf.constant([2.0, 3.0], name = "b")
    #result = a + b
    result = tf.add(a, b, name = "add")
    print(result)

def B():
    pass




if __name__ == "__main__":
    A()