from numpy import *
import matplotlib.pyplot as plt


# 读取文件数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 画图
def plotShow(xMat, yMat, yHat):
    '''
    :param xMat: type of xMat: <class 'numpy.matrixlib.defmatrix.matrix'>
    :param yMat: type of yMat: <class 'numpy.matrixlib.defmatrix.matrix'>
    :param yHat: type of yHat: <class 'numpy.ndarray'>
    :return:
    '''
    '''
    print("\n\ntype of xMat: %s" % type(xMat))
    print("xMat: %s" % xMat)
    print("\n\ntype of yMat: %s" % type(yMat))
    print("yMat: %s" % yMat)
    print("\n\ntype of yHat: %s" % type(yHat))
    print("yHat: %s" % yHat)
    '''

    srtInd = xMat[:, 1].argsort(0)  # x坐标排序后的索引
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s=2, c='red')
    plt.show()


# 画图，图中包括数据点和拟合直线
# xMat：测试数据横坐标，yMat测试数据纵坐标（即标签），xCopy：测试数据排序后横坐标, yHat：回归值
def draw(xMat, yMat, xCopy, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')  # 散点图
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


# 计算最佳拟合直线参数，该方法容易存在欠拟合问题
# 最小二乘法，找到直线ws，是的所有点到这条直线的距离平方和最小；对这个平方和求导，令导数为0，求得直线
# y = ws[0] + ws[1]*X1 + ws[2]*X2 + ......
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("该矩阵的行列式为0，无法求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# Locally Weighted Linear Regression
# 对于一个测试点，将他附近的点权重加大，得到回归值
# 这里的K越大，越容易欠拟合，越小越容易过拟合，对于ex0.txt的数据，k=0.01较为合适
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 将权重矩阵初始化为对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # 核函数
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("该矩阵的行列式为0，无法求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(xTx) == 0.0:
        print("该矩阵的行列式为0，无法求逆")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    # 数据标准化
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # python: mean()矩阵求取均值 -- numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue at 0x40b6a26c>)
    # 经常操作的参数为axis,以m * n矩阵举例:
    # axis 不设置值, 对 m*n 个数求均值, 返回一个实数
    # axis = 0: 压缩行, 对各列求均值, 返回 1* n 矩阵
    # axis =1: 压缩列, 对各行求均值, 返回 m *1 矩阵
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# 将所有点回归一遍
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def standRegresDraw():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)

    print("线性回归参数:\n %s" % ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws  # ws：线性回归参数，拟合直线估算的y值
    m, n = shape(yHat)
    yHatArr = zeros(m)
    for i in range(m):  # 把matrix转换成array
        yHatArr[i] = yHat[i][0]
    plotShow(xMat, yMat, yHatArr)


def lwlrTestDraw():
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    yMat = mat(yArr)
    plotShow(xMat, yMat, yHat)


# 鲍鱼数据测试
def abaloneTest():
    abX, abY = loadDataSet('abalone.txt')
    yHatArr01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHatArr1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHatArr10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)


def abaloneRidgeRegresTest():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


'''
standRegresDraw()
lwlrTestDraw()
abaloneTest()
abaloneRidgeRegresTest()
'''
