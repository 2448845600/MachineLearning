# 通过tensorflow实现一个两层的神经网络，目的是实现一个二次函数的拟合

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 增加一层神经网络，输出结果
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 注意该函数中是xW+b，而不是Wx+b。所以要注意乘法的顺序。x应该定义为[类别数量， 数据数量]， W定义为[数据类别，类别数量]。
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 矩阵乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def product_data():
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)  # noise函数为添加噪声所用，这样二次函数的点不会与二次函数曲线完全重合。
    y_data = np.square(x_data) - 0.5 + noise
    return x_data, noise, y_data


# 计算训练阈值
def calculate_train_step(data, prediction):
    ##
    # the error between prediciton and real data
    # tf.reduce_mean 求平均值
    # tf.reduce_sum 求和
    # tf.square 平方
    # reduction_indices 维度处理（如降维）
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - prediction), reduction_indices=[1]))
    # tf.train.GradientDescentOptimizer 梯度下降算法的优化器
    # minimize 通过更新var_list来减小loss，是compute_gradients() 和apply_gradients()的结合
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    return train_step


def main():
    x_data, noise, y_data = product_data()

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # add hidden layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, activation_function=None)

    train_step = calculate_train_step(ys, prediction)

    # 变量初始化
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # 画布为一行一列一块
    ax.scatter(x_data, y_data)
    plt.ion()  # 开启交互模式，show以后不暂停
    plt.show()

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # feed_dict 传入序列
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])  # 新增一条线之后去除原来的线
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # 可视化的连续线：x值，y值，红色，宽度5
            plt.pause(0.2)  # 每循环一次暂停0.2秒


main()
