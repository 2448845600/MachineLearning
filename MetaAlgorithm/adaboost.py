# --utf-8--
# http://blog.csdn.net/u011475210/article/details/78254505
from numpy import *


def loadSimpData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 比较阈值，对数据进行分类
# dataMatrix：数据集；dimen：数据集列数；threshVal：阈值；threshIneq：比较方式(根据阈值进行分类，'lt'表示小于，'gt'表示大于)
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # python: ones()
    #
    retArray = ones((shape(dataMatrix)[0], 1))  # 新建一个数组用于存放分类结果，初始化都为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 生成最低错误率的单层决策树
# D:权重向量
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T

    m, n = shape(dataMatrix)  # 获取行列值,同时n也就是属性的数量
    numSteps = 10.0  # 初始化步数，用于在特征的所有可能值上进行遍历
    bestStump = {}  # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息(dim, thresh, ineq)
    bestClassEst = mat(zeros((m, 1)))  # 初始化类别估计值
    minError = inf
    # 遍历各个属性
    for i in range(n):
        # python: numpy 数组基础操作 http://blog.csdn.net/mokeding/article/details/17476979
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 根据步数求得步长
        # 遍历不长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 遍历每个不等号
                threshVal = (rangeMin + float(j) * stepSize)  # 设定阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 通过阈值比较对数据进行分类
                errArr = mat(ones((m, 1)))  # 初始化错误计数向量
                errArr[predictedVals == labelMat] = 0  # 如果预测结果和标签相同，则相应位置0
                weightedError = D.T * errArr  # 计算权值误差，这就是AdaBoost和分类器交互的地方
                # print("split:dim %d, thresh %.2f, then ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                    minError = weightedError
                    # python: copy() deepcopy()
                    # b = a.copy(), a变化，b也变化；b = a.deepcopy()，a变化，b不变化
                    # http://blog.csdn.net/qq_32907349/article/details/52190796
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i  # 属性编号
                    bestStump['thresh'] = threshVal  # 阈值
                    bestStump['ineq'] = inequal  # 不等号
    return bestStump, minError, bestClassEst


# 训练adaBoost, numIt:最大循环次数，也就是弱分类器的个数，太大可能会过拟合
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 初始化向量D每个值均为1/m，D包含每个数据点的权重, 向量各项和为1
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print(D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        # print(classEst.T)
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst  # 累加类别估计值
        # print(aggClassEst.T)
        # 计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("第%d次 训练错误率: %f" % (i, errorRate))
        if errorRate == 0.0: break
    return weakClassArr


# 分类器
def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)


# 加载数据
def loadDataSet(fileName):
    # 自动获取特征个数，这是和之前不一样的地方
    numFeat = len(open(fileName).readline().split('\t'))
    # 初始化数据集和标签列表
    dataMat = [];
    labelMat = []
    # 打开文件
    fr = open(fileName)
    # 遍历每一行
    for line in fr.readlines():
        # 初始化列表，用来存储每一行的数据
        lineArr = []
        # 切分文本
        curLine = line.strip().split('\t')
        # 遍历每一个特征，某人最后一列为标签
        for i in range(numFeat - 1):
            # 将切分的文本全部加入行列表中
            lineArr.append(float(curLine[i]))
        # 将每个行列表加入到数据集中
        dataMat.append(lineArr)
        # 将每个标签加入标签列表中
        labelMat.append(float(curLine[-1]))
    # 返回数据集和标签列表
    return dataMat, labelMat


'''
datMat, classLabels = loadSimpData()
classifierArr = adaBoostTrainDS(datMat, classLabels, 9)
aggClassEstSign = adaClassify([0, 0], classifierArr)
print(aggClassEstSign)
'''


# 对horseColic数据集进行预测
def horseColic():
    print("训练集，每次循环训练集的错误率如下：")
    numIt = 10
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(dataArr, labelArr, numIt)
    print("%d次循环训练结果: %s" % (numIt, classifierArray))

    print("\n测试集，利用最后一次训练结果进行分类")
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = mat(ones((67, 1)))
    errNum = errArr[prediction10 != mat(testLabelArr).T].sum()
    errRate = float(errNum / 67)
    print("测试集错误率：%f" % errRate)


horseColic()
