# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0506_image_recognition.py
@Version    :   v0.1
@Time       :   2019-11-04 15:48
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0506，P105
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 图像识别
"""
import os
import sys
import sklearn
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tools import show_values
# import random
# from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

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

# MNIST Digit Prediction with k-Nearest Neighbors
# -----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x28 matrices
# of floating point numbers

# Load the data
# 对预测值进行了One-Hot编码
mnist = input_data.read_data_sets("../Data/MNIST_data/", one_hot = True)

# Random sample
train_size = 1000
test_size = 102
# np.random.choice(a,size,replace)：a是随机选择数的大小，size是随机数的个数，replace表示数据是否可以重复
rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace = False)
rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace = False)
x_vals_train = mnist.train.images[rand_train_indices]
x_vals_test = mnist.test.images[rand_test_indices]
y_vals_train = mnist.train.labels[rand_train_indices]
y_vals_test = mnist.test.labels[rand_test_indices]

# Declare k-value and batch size
# K近邻算法采用的是最为原始的方法，就是一个点与所有的点进行距离计算，然后找出距离最近的k个点，
# 再将k个点中的目标值统计个数得到最多目标值为指定数据的目标值。
# 因为数据量比较大，所以k值比4大，效果要好些。
# k近邻算法的随机性比较强，精确度不太稳定
k = 7
batch_size = 6

# Placeholders
# 784 = 28 * 28 的图片；10 表示 10 个数字
x_data_train = tf.placeholder(shape = [None, 784], dtype = tf.float32)
x_data_test = tf.placeholder(shape = [None, 784], dtype = tf.float32)
y_target_train = tf.placeholder(shape = [None, 10], dtype = tf.float32)
y_target_test = tf.placeholder(shape = [None, 10], dtype = tf.float32)

# Declare distance metric
# L2 范数的精确度是 5%，L1 范数的精确度是 83%
# L1 norm
# reduction：数据归约
# 1000个训练数据集中的数据点与测试数据集中取出的6个数据点分别相减，形成一个[6,1000,784]维度的数据。
tf_abs = tf.abs(x_data_train - tf.expand_dims(x_data_test, 1))
# 将第3维数据归约加（即784个图像点的差相加），形成一个[6,1000]的数据，这个结果就是6个数据点与训练数据集中数据点的距离
distance = tf.reduce_sum(tf_abs, reduction_indices = 2)

# L2 norm
# 因为图像采用平方距离度量只会加强相似性的要求，使得图像处理中对于图像变形的情况无法很好地适应
# distance = tf.sqrt(tf.reduce_sum(tf.square(x_data_train - tf.expand_dims(x_data_test, 1)), reduction_indices = 1))

# Predict: Get min distance index (Nearest neighbor)
# 最近邻算法是聚类算法（无监督的），不需要先训练，再预测的。
top_k_xvals, top_k_indices = tf.nn.top_k(-distance, k = k)
prediction_indices = tf.gather(y_target_train, top_k_indices)
# Predict the mode category
count_of_predictions = tf.reduce_sum(prediction_indices, reduction_indices = 1)
prediction = tf.argmax(count_of_predictions, dimension = 1)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test) / batch_size))

test_output, actual_vals, x_batch, y_batch, predictions = [], [], [], [], []
for i in range(num_loops):
    min_index = i * batch_size
    max_index = min((i + 1) * batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    feed_dict = {
            x_data_train: x_vals_train,
            x_data_test: x_batch,
            y_target_train: y_vals_train,
            y_target_test: y_batch
    }
    # show_values(x_data_train, session = sess, feed_dict = {x_data_train: x_vals_train})
    # show_values(x_data_test, session = sess, feed_dict = {x_data_test: x_batch})
    # show_values(distance, session = sess, feed_dict = {x_data_train: x_vals_train, x_data_test: x_batch})
    predictions = sess.run(prediction, feed_dict = feed_dict)
    test_output.extend(predictions)
    actual_vals.extend(np.argmax(y_batch, axis = 1))
    pass

accuracy = sum([1. / test_size for i in range(test_size) if test_output[i] == actual_vals[i]])
print("Accuracy on test set: ", accuracy)

# Plot the last batch results:
actuals = np.argmax(y_batch, axis = 1)

Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i + 1)
    plt.imshow(np.reshape(x_batch[i], [28, 28]), cmap = 'Greys_r')
    plt.title("真实值：" + str(actuals[i]) + "-----" + "预测值：" + str(predictions[i]), fontsize = 10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    pass
plt.suptitle("图5-4：最近领域算法预测的最后批次的六张图片")

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
