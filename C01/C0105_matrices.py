# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0105_matrices.py
@Version    :   v0.1
@Time       :   2019-10-29 11:03
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0105，P7
@Desc       :   TensorFlow 基础，操作矩阵
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


# 1.5 矩阵
number_title = "TensorFlow 声明矩阵"
print('\n', '-' * 5, number_title, '-' * 5)
# Identity matrix
identity_matrix = tf.diag([1.0, 1.0, 1.0])
show_values(identity_matrix, "identity_matrix")


# a 12x13 truncated random normal distribution
A = tf.truncated_normal(shape =[12, 13] )
show_values(A, "A")

# 12x13 constant matrix, fill matrix with 5.0
B = tf.fill([12, 13], 5.0)
show_values(B, "B")

# a 13x12 random uniform distribution
C = tf.random_uniform(shape =[13, 12])
show_values(C, "C")
show_values(C, "Rerun C")

# 将np矩阵转换为张量
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
show_values(D, "D")

show_values(A + B, "A+B")
show_values(A - B, "A-B")
show_values(tf.matmul(A, C), "A*C")
show_values(tf.transpose(A), "A'")
show_values(tf.matrix_determinant(D), "|D|")
show_values(tf.matrix_inverse(D), "D^(-1)")
show_values(tf.cholesky(identity_matrix), "cholesky(identity_matrix)")
# cholesky()分解，必须是对称正定阵
# show_values("cholesky(D)", tf.cholesky(D))
# 第一行是特征值
# 剩下的是特征向量
show_values(tf.self_adjoint_eig(D), "eigen_values + eigen vectors")

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
