import tensorflow as tf 

with tf.variable_scope("foo"):
    #在命名空间foo下获取变量bar，于是得到的变量名称为“foo/bar”
    a = tf.get_variable("bar",shape=[1])
    print(a.name)   #输出：foo/bar:0

with tf.variable_scope("bar"):
    #在命名空间bar下获取变量"bar",于是得到的变量名称为“bar/bar”。此时变量
    #"bar/bar"和变量“foo/bar”并不冲突，于是可以正常运行
    b = tf.get_variable("bar",shape=[1])
    print(b.name)   #bar/bar:0

with tf.name_scope("a"):
    #使用tf.variable函数生成的变量会受tf.name_scope影响，于是这个变量的名称为
    #“a/Variable"
    a = tf.Variable([1])
    b = tf.Variable([1])
    print(a.name,b.name)   #a/Variable:0,a/Variable_1:0

    #tf.get_variable函数不受tf.name_scope函数的影响
    #于是变量并不在a这个命名空间里面
    b = tf.get_variable("b",shape=[1])
    print(b.name)   #b:0

with tf.name_scope("b"):
    #因为tf.get_variable不受tf.name_scope影响，所以这里将试图获取
    #名称为“a”的变量。然而这个变量已经被声明了，于是这里会报错
    tf.get_variable("b",[1])# Variable b already exists, disallowed. 
    

