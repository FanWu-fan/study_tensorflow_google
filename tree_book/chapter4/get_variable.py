import tensorflow as tf 

with tf.variable_scope("foo"):
    v = tf.get_variable(
        "v",[1],initializer=tf.constant_initializer(1.0))

#因为在命名空间foo中已经存在名字为v的变量，所有下面的代码将会报错：
# with tf.variable_scope("foo"):
#     v = tf.get_variable(
#         "v",[1]
#     )

#在生成上下文管理器时，将参数reuse设置为True,这样 tf.get_variable
#函数将直接获取已经声明的变量。
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v",[1])
    print (v == v1) #输出为True,代表v，v1代表的时相同的TF变量

#将参数reuse 设置为True时，tf.variable_scope
#将只能获取已经创建过的变量，因为在命名空间bar中还没有创建
#变量v，所以下面的代码将会报错
# with tf.variable_scope("bar",reuse=True):
#     v= tf.get_variable("v",[1])

#嵌套
with tf.variable_scope("root"):
    #可以通过tf.get_variable_scope().reuse函数来获取当前上下文
    #管理器中reuse参数的取值
    print(tf.get_variable_scope().reuse)#输出False,即最外层reuse是Fasle

    with tf.variable_scope("foo",reuse=True):
        print(tf.get_variable_scope().reuse)#True
        
        #新建嵌套的上下文管理器，不指定reuse，reuse取值会和外面一层一致
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)#True

v1 = tf.get_variable("v",[1])
print(v1.name) #输出v:0,“v”为变量的名称，“0”表示这个变量是生成变量这个
#运算的第一个结果

v2 = tf.get_variable("v1",[1])
print(v2.name)#输出v1:0,“v”为变量的名称，“0”表示这个变量是生成变量这个
#运算的第一个结果

with tf.variable_scope("new"):
    v2 = tf.get_variable("v",[1])
    print(v2.name)#输出new/v:0.在tf.variable_scope中创建的变量，名称
    #前面会加入命名空间的名称，并通过/来分隔命名空间的名称和变量的名称

with tf.variable_scope("new"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        print(v3.name)#输出new/bar/v:0.

    v4 = tf.get_variable("v1",[1])
    print(v4.name)#new/v1:0

#创建一个名称为空的命名空间。并设置 reuse=True
with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("new/bar/v",[1])
    print(v5==v3)#True

    v6= tf.get_variable("new/v1",[1])
    print(v6==v4)#True
    