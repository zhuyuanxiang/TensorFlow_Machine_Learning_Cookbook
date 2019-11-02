# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   tools.py
@Version    :   v0.1
@Time       :   2019-11-02 11:58
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec04，P
@Desc       :   基于 TensorFlow 的线性回归，常用的 Python 工具函数
"""
# Common imports
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import pandas as pd

# 设置数据显示的精确度为小数点后7位
np.set_printoptions(precision = 7, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)

import os
import sys
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

import winsound

# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass


def show_values(var_name, variable, feed_dict = None, session = None):
    if session is None:
        session = tf.Session()
    print('-' * 50)
    print("{} = {}".format(var_name, variable))
    print("session.run({}) = ".format(var_name))
    result = session.run(variable, feed_dict = feed_dict)
    print(result)
    return result


def others():
    tmp_4_shape_1 = tf.constant(np.arange(0, 16, dtype = np.int32), shape = [4, 1, 1, 4])
    tmp_4_shape_2 = tf.constant(np.arange(0, 16, dtype = np.int32), shape = [4, 1, 4, 1])
    tmp_4_cross = tf.matmul(tmp_4_shape_1, tmp_4_shape_2)
    tmp_4_cross = tf.matmul(tmp_4_shape_2, tmp_4_shape_1)
    sess.run(tmp_4_cross)