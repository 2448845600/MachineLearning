# --utf-8--
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 增加参数X0，并且设置为1
        # 线性模型有一个常数项b的，它代表了拟合线的上下浮动，增加一个参数X0就是为了求这个常数项b，为了方便起见，X0赋值为1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升，返回回归系数向量
def gradAscent(dataMatIn, classLabels):
    # 转换为numpy的矩阵，否则运算会出错
    # transpose()是转置
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001  # 梯度上升的距离(步长)
    maxCycles = 500  # 最大循环次数
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升的循环过程
        h = sigmoid(dataMatrix * weights)  # 这里是对所有数据的矩阵进行操作，表面上是一次乘法，实际上是300次乘法
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error  # 回归系数为原来的回归系数加上步长乘以数据集行列变换后乘以误差列向量，得w1,w2,w3三个回归系数
    # 这里weights是一个矩阵[[ 4.12414349][ 0.48007329][-0.6168482 ]]，要利用getA()转换为array
    return weights.getA()


# 随机梯度上升
# 随机梯度上升与梯度上升在代码上很相似，但是也有一些区别：1 后者的变量h和误差error都是向量而前者则全是数值 2 前者矩阵变换过程，所有变量的数据类型都是numPy数组
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    # print(weights)
    # 这里weights是一个array[ 1.01702007  0.85914348 -0.36579921]
    return weights


# 随机梯度上升，改进版1.0
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    print(m, n)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (list(dataIndex)[randIndex])  # 每次i循环的时候，随机不重复，防止每次j循环之间的周期性波动
    # print(weights)
    # 这里weights是一个array，每次不一样
    return weights


# 画图
def plotBestFit(wei):
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()


# 测试logistic的性能
def test():
    dataMat, labelMat = loadDataSet()
    # 梯度上升
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
    # 随机梯度上升
    weights0 = stocGradAscent0(array(dataMat), labelMat)
    plotBestFit(weights0)
    # 随机梯度上升，改进版1.0
    weights1 = stocGradAscent1(array(dataMat), labelMat)
    plotBestFit(weights1)

# test()
