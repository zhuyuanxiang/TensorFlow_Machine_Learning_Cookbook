# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0204_multiple_layers.py
@Version    :   v0.1
@Time       :   2019-10-29 14:38
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0204，P23
@Desc       :   TensorFlow 进阶，TensorFlow 的多层 Layer
"""
import os
import sys
import sklearn
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tools import show_values

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf,
                    linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()


# 2.4 多层
x_shape = [1, 4, 4, 1]
x_vals = np.random.uniform(size = x_shape)
# x_vals = np.array([x_vals, x_vals +1])
print("x_vals = ")
print(x_vals)

x_data = tf.placeholder(tf.float32, shape = x_shape)

# filter 滤波器
my_filter = tf.constant(0.25, shape = [2, 2, 1, 1])
show_values(my_filter, "my_filter")
# stride 步长
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding = 'SAME', name = 'Moving_Avg_Window')


# Define a custom layer which will be sigmoid(Ax+b) where
# x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
    # tf.squeeze() 删除所有一维的维度
    # temp_tsr=tf.constant([[2,3]])
    # sess.run(tf.squeeze(temp_tsr)) --> [2,3]
    # temp_tsr = tf.constant([[[[[[2]]], [[[3]]], [[[4]]]]], [[[[[5]]], [[[6]]], [[[7]]]]]])
    # sess.run(tf.squeeze(temp_tsr)) --> [[2,3,4],[5,6,7]]
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape = [2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    return tf.sigmoid(temp)


# 使用命名域管理复杂的计算图
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)
    pass

show_values(custom_layer1, "custom_layer(mov_avg_layer) = ",
            feed_dict = {x_data: x_vals})

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
