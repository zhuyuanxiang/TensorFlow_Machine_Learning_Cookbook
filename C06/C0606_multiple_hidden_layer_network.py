# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0606_multiple_hidden_layer_network.py
@Version    :   v0.1
@Time       :   2019-11-07 11:28
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0606，P126
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 多层神经网络
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
seed = 42
tf.set_random_seed(seed)
np.random.seed(seed)

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

# The 'Low Birthrate Dataset' is a dataset provided by Univ. of Massachusetts at Amherst.
# It is a great dataset used for numerical prediction (birthweight)
# and logistic regression (binary classification, birthweight`<`2500g or not).
# Information about it is located here:",
# 这个地址的数据不允许访问，本地数据集中只有9维数据，缺少“FTV”数据
# birth_data_url = "https://www.umass.edu/statdata/statdata/data/lowbwt.txt"
# birth_file = requests.get(birth_data_url)
# birth_data = birth_file.text.split('\r\n')[5:]
with open("../Data/birthweight_data/birthweight.dat") as f:
    birth_file = f.read()
    pass
birth_data = birth_file.split('\n')
birth_header = [x for x in birth_data[0].split('\t') if len(x) >= 1]
# Interesting : 下面这个用法很有趣，值得关注
birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
batch_size = 100

# Extract y-traget(birth weight)
y_vals = np.array([x[8] for x in birth_data])
# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest]
                  for x in birth_data])
# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_min) / (col_max - col_min)

# ToDo:分别归一化的方法是错的
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev = st_dev))
    return weight


def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev = st_dev))
    return bias


# Create Placeholders
x_data = tf.placeholder(shape = [None, 7], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = input_layer @ weights + biases
    return tf.nn.relu(layer)


# --------Create the first layer (25 hidden nodes)--------
weight_1 = init_weight(shape = [7, 25], st_dev = 10.0)
bias_1 = init_bias(shape = [25], st_dev = 10.0)
layer_1 = fully_connected(x_data, weight_1, bias_1)

# --------Create second layer (10 hidden nodes)--------
weight_2 = init_weight(shape = [25, 10], st_dev = 10.0)
bias_2 = init_bias(shape = [10], st_dev = 10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# --------Create third layer (3 hidden nodes)--------
weight_3 = init_weight(shape = [10, 3], st_dev = 10.0)
bias_3 = init_bias(shape = [3], st_dev = 10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# --------Create output layer (1 output value)--------
weight_4 = init_weight(shape = [3, 1], st_dev = 10.0)
bias_4 = init_bias(shape = [1], st_dev = 10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# Declare loss function (L1)
loss = tf.reduce_mean(tf.abs(y_target - final_output))

# Declare optimizer
# 《深度学习》第8章
# https://www.jianshu.com/p/e6e8aa3169ca
# https://blog.csdn.net/willduan1/article/details/78070086
# https://blog.csdn.net/TeFuirnever/article/details/88933368
# 参数：
# learning_rate：张量或浮点值。学习速率
# beta1：一个浮点值或一个常量浮点张量。一阶矩估计的指数衰减率
# beta2：一个浮点值或一个常量浮点张量。二阶矩估计的指数衰减率
# epsilon：数值稳定性的一个小常数
# use_locking：如果True，要使用lock进行更新操作
# `name``：应用梯度时为了创建操作的可选名称。默认为“Adam”
# 多层感知机的性能不稳定，参数设置非常重要
# 可能是数据维度不够，训练的精度不如原书中的好
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i + 1) % 250 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

# Plot loss over time
plt.plot(loss_vec, 'b-', label = 'Train Loss')
plt.plot(test_loss, 'r--', label = 'Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc = "upper right")

# Find the % classified correctly above/below the cutoff of 2500 g
# >= 2500 g = 0
# < 2500 g = 1
actuals = np.array([x[0] for x in birth_data])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]

test_preds = [x[0] for x in sess.run(final_output, feed_dict = {x_data: x_vals_test})]
train_preds = [x[0] for x in sess.run(final_output, feed_dict = {x_data: x_vals_train})]
test_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in test_preds])
train_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in train_preds])

# Print out accuracies
test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)])
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
