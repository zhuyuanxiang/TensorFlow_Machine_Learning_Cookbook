# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0209_evaluating_models.py
@Version    :   v0.1
@Time       :   2019-10-30 11:20
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0209，P40
@Desc       :   TensorFlow 进阶，TensorFlow 实现模型评估
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


# 回归模型。
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.
def regression_model():
    # Declare batch size
    batch_size = 25

    # Create data
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
    y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

    # Split data into train/test = 80%/20%
    train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * .8)),
                                     replace = False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    # Create variable (one model parameter = A)
    A = tf.Variable(tf.random_normal(shape = [1, 1]))

    # Add operation to graph
    my_output = tf.matmul(x_data, A)

    # Add L2 loss operation to graph
    loss = tf.reduce_mean(tf.square(my_output - y_target))

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create Optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    # Run Loop
    for i in range(100):
        rand_index = np.random.choice(len(x_vals_train), size = batch_size)
        rand_x = np.transpose([x_vals_train[rand_index]])
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
        if i % 11 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict = {
                    x_data: rand_x, y_target: rand_y
            })))

    # Evaluate accuracy (loss) on test set
    mse_test = sess.run(
            loss, feed_dict = {
                    x_data: np.transpose([x_vals_test]),
                    y_target: np.transpose([y_vals_test])
            })
    mse_train = sess.run(
            loss, feed_dict = {
                    x_data: np.transpose([x_vals_train]),
                    y_target: np.transpose([y_vals_train])
            })
    print("MSE on test:" + str(np.round(mse_test, 2)))
    print("MSE on train:" + str(np.round(mse_train, 2)))


# 分类模型。
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2
def classification_model():
    # Declare batch size
    batch_size = 250

    # Create data
    x_vals = np.concatenate(
            (np.random.normal(-1, 1, 500), np.random.normal(2, 1, 500)))
    y_vals = np.concatenate((np.repeat(0., 500), np.repeat(1., 500)))
    x_data = tf.placeholder(shape = [1, None], dtype = tf.float32)
    y_target = tf.placeholder(shape = [1, None], dtype = tf.float32)

    # Split data into train/test = 80%/20%
    train_indices = np.random.choice(len(x_vals),
                                     int(round(len(x_vals) * 0.8)),
                                     replace = False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    # Create variable (one model parameter = A)
    A = tf.Variable(tf.random_normal(mean = 10, shape = [1]))

    # Add operation to graph
    # Want to create the operstion sigmoid(x + A)
    # Note, the sigmoid() part is in the loss function
    my_output = tf.add(x_data, A)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Add classification loss (cross entropy)
    xentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = my_output,
                                                    labels = y_target))

    # Create Optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)

    # Run loop
    for i in range(2800):
        rand_index = np.random.choice(len(x_vals_train), size = batch_size)
        rand_x = [x_vals_train[rand_index]]
        rand_y = [y_vals_train[rand_index]]
        sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
        if (i + 1) % 200 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(xentropy, feed_dict = {
                    x_data: rand_x, y_target: rand_y
            })))

    # Evaluate Predictions on test set
    y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
    correct_prediction = tf.equal(y_prediction, y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_value_train = sess.run(accuracy, feed_dict = {
            x_data: [x_vals_train], y_target: [y_vals_train]
    })
    acc_value_test = sess.run(accuracy, feed_dict = {
            x_data: [x_vals_test], y_target: [y_vals_test]
    })
    print("Accuracy on train set: " + str(acc_value_train))
    print("Accuracy on test set: " + str(acc_value_test))

    # Plot classification result
    A_result = - sess.run(A)
    bins = np.linspace(-5, 5, 500)
    plt.hist(x_vals[0:500], bins, alpha = 0.5, label = 'N(-1,1)',
             color = 'blue')
    plt.hist(x_vals[500:1000], bins, alpha = 0.5, label = 'N(2,1)',
             color = 'red')
    plt.plot((A_result, A_result), (0, 8), 'k--', linewidth = 3,
             label = 'A = ' + str(np.round(A_result, 2)))
    plt.legend(loc = 'upper right')
    plt.suptitle("图2-8：模型A和数据点的可视化——两个正态分布（均值为-1和2）。\n"
                 "理论上的最佳分割点是(0.5)，模型结果为（{}）。".format(A_result))


if __name__ == "__main__":
    # regression_model()
    classification_model()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
