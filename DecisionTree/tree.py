# --utf8--
from math import log
import operator
from DecisionTree.treePlotter import createPlot


# 计算该数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
# axis划分数据集的特征(key)，value特征的返回值
# 将数据集按照特征axis的值是否等于给定值value进行划分，返回的是不包含axis属性的axis属性不为value的“加工后”的数据集（有二分的意思）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 将数据的axis列删除，将featVec[axis] != value的行删除
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式，即按照此属性划分数据集后，信息增益最大，返回该属性的序号
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1#数据集的最后一项不是属性
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature


# 如果数据集已经处理了所有属性，但类标签依然不唯一，此时程序需要决定如何定义该叶子节点，这里采用多数表决的方法决定该叶子节点的分类
def majortyCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归建立决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同时，停止划分，返回该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征后，停止划分，返回多数表决后的标签
    if len(dataSet[0]) == 1:
        return majortyCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #print("labels:" + label for label in labels)
        subLabels = labels[:]
        #print("subLabels:" + subLabel for subLabel in subLabels)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 判断secondDict[key]是否为字典
            # 若为字典，递归的寻找testVec
            if type(secondDict[key])._name_ == 'dict':classLabel = classify(secondDict[key], featLabels, testVec)
            # 若secondDict[key]为标签值，则将secondDict[key]赋给classLabel（就是要一直找到叶子结点）
            else: classLabel = secondDict[key]
    return classLabel


# 序列化，数据以二进制存放在.pkl文件中
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()


# 反序列化，二进制打开.pkl文件
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def test():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    storeTree(lensesTree, "lensesTreeStore.pkl")
    grabTree("lensesTreeStore.pkl")
    #createPlot(lensesTree)

def readFromFile():
    lensesTree = grabTree("lensesTreeStore.pkl")
    createPlot(lensesTree)


test()
print("序列化反序列化")
readFromFile()


