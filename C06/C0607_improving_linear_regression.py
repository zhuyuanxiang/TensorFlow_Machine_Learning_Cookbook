# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0607_improving_linear_regression.py
@Version    :   v0.1
@Time       :   2019-11-07 16:25
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0607，P131
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 优化的线性预测模型
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
np.random.seed(seed)
tf.set_random_seed(seed)
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

# Pull out target variable
y_vals = np.array([x[0] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([x[1:8] for x in birth_data])

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare batch size
batch_size = 90

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 7], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)


# Create variable definition
def init_variable(shape):
    return tf.Variable(tf.random_normal(shape = shape))


# Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = input_layer @ multiplication_weight + bias_weight
    # We separate the activation at the end because the loss function will
    # implement the last sigmoid necessary
    if activation:
        return tf.nn.sigmoid(linear_layer)
    else:
        return linear_layer


# First logistic layer (7 inputs to 14 hidden nodes)
A1 = init_variable(shape = [7, 14])
b1 = init_variable(shape = [14])
logistic_layer1 = logistic(x_data, A1, b1)

# Second logistic layer (14 hidden inputs to 5 hidden nodes)
A2 = init_variable(shape = [14, 7])
b2 = init_variable(shape = [7])
logistic_layer2 = logistic(logistic_layer1, A2, b2)

# Final output layer (5 hidden nodes to 1 output)
A3 = init_variable(shape = [7, 3])
b3 = init_variable(shape = [3])
logistic_layer3 = logistic(logistic_layer2, A3, b3)
A4 = init_variable(shape = [3, 1])
b4 = init_variable(shape = [1])
final_output = logistic(logistic_layer3, A4, b4, activation = False)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target, logits = final_output))

# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.003)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Actual Prediction
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Training loop
loss_vec = []
train_acc = []
test_acc = []
for i in range(15000):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    # loss_vec.append(temp_loss)

    temp_acc_train = sess.run(accuracy, feed_dict = {x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    # train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    # test_acc.append(temp_acc_test)
    if (i + 1) % 250 == 0:
        loss_vec.append(temp_loss)
        train_acc.append(temp_acc_train)
        test_acc.append(temp_acc_test)
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))
# 训练的效果并不太好
# 分析原因：可能是特征与预测结果的关联性并非正确；可能是模型设计的不太合理；可能数据量不够丰富，训练次数过多后会造成过拟合
# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'b-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.ylim([0, 2])

# Plot train and test accuracy
plt.figure()
plt.plot(train_acc, 'b-', label = 'Train Set Accuracy')
plt.plot(test_acc, 'r--', label = 'Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')

if len(plt.get_fignums()) != 0:
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    plt.show()
pass
