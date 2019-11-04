# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0103_tensors.py
@Version    :   v0.1
@Time       :   2019-10-29 11:18
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0103，P3
@Desc       :   TensorFlow 基础，张量
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
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)
# to make this notebook's output stable across runs
np.random.seed(42)

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


# 1.3 声明张量
def declare_fix_tensor():
    # 1. 固定张量
    row_dim, col_dim = (3, 2)
    # 创建指定维度的零张量
    # Zero initialized variable
    zeros_tsr = tf.zeros([row_dim, col_dim])
    with sess.as_default():
        print(zeros_tsr.eval())
    show_values(zeros_tsr, 'zeros_tsr')

    # 创建指定维度的单位张量
    # One initialized variable
    ones_tsr = tf.ones([row_dim, col_dim])
    show_values(ones_tsr, "ones_tsr")

    # 创建指定维度的常数填充的张量
    filled_tsr = tf.fill([row_dim, col_dim], 42)
    show_values(filled_tsr, "filled_tsr")

    # 创建常数张量
    # Fill shape with a constant
    constant_tsr = tf.constant([[1, 2, 3], [4, 5, 6]])
    show_values(constant_tsr, "constant_tsr")

    # Create a variable from a constant
    const_tsr = tf.constant([8, 6, 7, 5, 3, 0, 9])
    show_values(const_tsr, "const_tsr")
    # This can also be used to fill an array:
    const_fill_tsr = tf.constant(-1, shape = [row_dim, col_dim])
    show_values(const_fill_tsr, "const_fill_tsr")
    pass


# 2. 相似形状的张量
# shaped like other variable
def declare_similar_tensor():
    number_title = "TensorFlow 声明相似形状的张量"
    print('\n', '-' * 5, number_title, '-' * 5)

    constant_tsr = tf.constant([[1, 2, 3], [4, 5, 6]])

    # 相似形状的零张量
    zeros_similar_tsr = tf.zeros_like(constant_tsr)
    show_values(zeros_similar_tsr, "zeros_similar_tsr")

    # 相似形状的单位张量
    ones_similar_tsr = tf.ones_like(constant_tsr)
    show_values(ones_similar_tsr, "ones_similar_tsr")

    # 运算符重载
    two_similar_tsr = ones_similar_tsr + ones_similar_tsr
    show_values(two_similar_tsr, "two_similar_tsr")

    four_similar_tsr = two_similar_tsr * two_similar_tsr
    show_values(four_similar_tsr, "four_similar_tsr")

    neg_four_similar_tsr = -four_similar_tsr
    show_values(neg_four_similar_tsr, "neg_four_similar_tsr")

    neg_eight_similar_tsr = 2 * neg_four_similar_tsr
    show_values(neg_eight_similar_tsr, "neg_eight_similar_tsr")

    eight_similar_tsr = abs(neg_eight_similar_tsr)
    show_values(eight_similar_tsr, "eight_similar_tsr")

    twelve_similar_tsr = eight_similar_tsr - neg_four_similar_tsr
    show_values(twelve_similar_tsr, "twelve_similar_tsr")

    three_similar_tsr = twelve_similar_tsr / four_similar_tsr
    show_values(three_similar_tsr, "three_similar_tsr")

    a = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype = tf.float64)
    a_one = a * a  # 这个是矩阵数乘，不是矩阵乘法
    show_values(a_one, "a_one")

    a_floor_div = a // three_similar_tsr
    show_values(a_floor_div, "a_floor_div")

    a_mod = a % three_similar_tsr
    show_values(a_mod, "a_mod")

    a_power = a ** three_similar_tsr
    show_values(a_power, "a_power")


# 3. 序列张量
def declare_seq_tensor():
    number_title = "TensorFlow 声明序列张量"
    print('\n', '-' * 5, number_title, '-' * 5)
    linear_seq_tsr = tf.linspace(start = 0.0, stop = 1.0, num = 3)
    show_values(linear_seq_tsr, "linear_seq_tsr")

    integer_seq_tsr = tf.range(start = 6, limit = 15, delta = 3)
    show_values(integer_seq_tsr, "integer_seq_tsr")


# 4. 随机张量
def declare_random_tensor():
    number_title = "TensorFlow 声明随机张量"
    print('\n', '-' * 5, number_title, '-' * 5)

    row_dim, col_dim = (13, 12)
    # 均匀分布的随机数
    randunif_tsr = tf.random_uniform([row_dim, col_dim], minval = 0, maxval = 1)
    show_values(randunif_tsr, "randunif_tsr")

    # 正态分布的随机数
    randnorm_tsr = tf.random_normal([row_dim, col_dim], mean = 0.0, stddev = 1.0)
    show_values(randnorm_tsr, "randnorm_tsr")

    # 带有指定边界的正态分布的随机数
    runcnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean = 0.0, stddev = 1.0)
    show_values(runcnorm_tsr, "runcnorm_tsr")

    # 张量随机化
    shuffled_output = tf.random_shuffle(randunif_tsr)
    show_values(shuffled_output, "shuffled_output")

    # 张量的随机剪裁
    cropped_output = tf.random_crop(randunif_tsr, [7, 5])
    show_values(cropped_output, "cropped_output")

    # 这个是剪裁图片的例子，没有图片，不能执行
    # cropped_image = tf.random_crop(my_image, [height / 2, width / 2, 3])
    # my_var = tf.Variable(tf.zeros([row_dim, col_dim]))


if __name__ == "__main__":
    #
    # declare_fix_tensor()

    declare_similar_tensor()
    #
    # declare_seq_tensor()

    # declare_random_tensor()
    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
