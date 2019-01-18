import tensorflow as tf
"""
定义位置： tensorflow/python/ops/variable_scope.py
此上下文管理器验证(可选)指来之同一个计算图，确保计算图是默认计算图，并且推送名称范围和变量范围。

如果name_orscope不是None，它按照原样使用，如果name_or _scope是None,则使用default_name,
在这种情况下，如果先前在同一范围内同一名字已经被使用，它将会被附加_N成为唯一的名字

variable_scope允许你创造一个新的变量并共享已经创建的变量，同时提供不会意外的创建或者共享的检查，
"""
#Simple example of how to creat a new variable
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
#Simple example of how to reenter a premade variable scope safely:
with tf.variable_scope("foo") as vs：
    pass
#Re-enter the variable scope
with tf.variable_scope(vs,
                        auxiliary_name_scope=False) as vs1:
    #Restore the original name_scope
    with tf.name_scope(vs1.original_name_scope):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/v:0"
        c = tf.constant([1], name="c")
        assert c.name == "foo/c:0"

# Basic example of sharing a variable AUTO_REUSE:
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2