# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0106_operations.py
@Version    :   v0.1
@Time       :   2019-10-29 14:11
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec01，P1
@Desc       :   TensorFlow 基础
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后3位
from tools import show_values

np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize = 14)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

import os
import sys
import sklearn
import tensorflow as tf
from tensorflow.python.framework import ops

# 初始化默认的计算图
ops.reset_default_graph()
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Open graph session
sess = tf.Session()

# 整数除
show_values(tf.div(3, 4), "tf.div(3,4)")
# 浮点除
show_values(tf.truediv(3, 4), "tf.truediv(3,4)")
# 浮点取整除
show_values(tf.floordiv(3.0, 4.0), "tf.floordiv(3.0,4.0)")
# 取模
show_values(tf.mod(22.0, 5.0), "tf.mod(22.0,5.0)")
# 张量点积--Compute the pairwise cross product，必须是三维向量
# 两个向量的叉乘，又叫向量积、外积、叉积，叉乘的运算结果是一个向量而不是一个标量。
# 两个向量的叉积与这两个向量组成的坐标平面垂直。
show_values(tf.cross([1., 0., 0.], [0., 1., 0.]),
            "tf.cross([1., 0., 0.], [0., 1., 0.])")
# show_values("tf.cross([1., 0., 0.,0.], [0., 1., 0.,0.])", tf.cross([1., 0., 0., 0.], [0., 1., 0., 0.]))

# P11，数学函数列表

show_values(tf.div(tf.sin(3.1416 / 4.), tf.cos(3.1416 / 4.)),
            "tan(pi/4) = 1 = tf.div(tf.sin(3.1416/4.),tf.cos(3.1416/4.))")


test_nums = range(15)

# 自定义函数
# 3x^2-x+10,x=11,=>
def custom_polynomial(value):
    return (tf.subtract(3 * tf.square(value), value) + 10)


show_values(custom_polynomial(11), "3x^2-x+10,x=11=>")

# What should we get with list comprehension
expected_output = [3 * x * x - x + 10 for x in test_nums]
print('-'*50)
print("[3 * x * x - x + 10 for x in test_nums] = ")
print(expected_output)

# Tensorflow custom function output
for num in test_nums:
    show_values(custom_polynomial(num), "custom_polynomial({})".format(num))

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
