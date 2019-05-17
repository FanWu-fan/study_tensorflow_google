import tensorflow as tf 
v1 = tf.get_variable("v1",[1],initializer=tf.constant_initializer(1.0))
v2 = tf.get_variable("v2",[1],initializer=tf.constant_initializer(2.0))
result1 = v1 + v2

saver = tf.train.Saver()
#通过export_meta_graph函数导出TF计算图的 元图，并保存为 json格式
saver.export_meta_graph("./save_2json_meta/model.ckpt.meda.json",as_text=True)