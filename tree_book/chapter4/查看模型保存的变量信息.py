import tensorflow as tf 

#tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader("./test_Saver/model.ckpt")

#获取变量名列表，，这个是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variables_name in all_variables:
    #variables_name 为变量名称，all_variables[variables_name]为变量的维度
    print(variables_name,all_variables[variables_name])

#获取名称为 v1的变量的取值
print("Value for variables v1 is ",reader.get_tensor("v1_tf"))

'''
v1_tf [1]
v2_tf [1]
Value for variables v1 is  [1.]
'''