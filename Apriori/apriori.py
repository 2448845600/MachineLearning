# 从数据集中挖掘频繁项（apriori），并且分析关联规则（generateRules）
# apriori算法核心概念：支持度(minSupport就是一个玄学参数)，候选项和频繁项
# 子集不为频繁项，则该集合不为频繁项
# 规则 P -> H，P 称为规则左部，H 称为规则右部


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    :param dataSet: 数据集
    :return: 长度为 1 的候选集集合
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    frozensetC1 = list(map(frozenset, C1))
    # print(frozensetC1+frozensetC1)
    return frozensetC1  # frozenset表示创建后不能改变


def scanD(D, ck, minSupport):
    """
    :param D: 数据集
    :param ck: k阶候选集合列表
    :param minSupport: 最小支持度
    :return: 符合条件的候选集（即频繁集） Lk，和所有候选项的支持度字典
    """
    ssCnt = {}  # 创建空字典
    for tid in D:
        for can in ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    Lk = []  # 符合支持度的候选项
    supportData = {}  # 支持度字典
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            Lk.insert(0, key)
        supportData[key] = support
    return Lk, supportData


def aprioriGetCk(Lk_1, k):
    """
    :param Lk_1: k-1阶频繁项列表
    :param k: k阶，所以k最小值为2，k-2 >= 0
    :return: k阶候选项列表
    """
    Ck = []
    lenLk_1 = len(Lk_1)

    # 对于Lk_1[]中的每一项，其长度为k-1，我们挑选前 k-2 项相同的合并，合并后长度为 k，且不重复，可以看一下机器学习实战P208的讲解
    for i in range(lenLk_1):
        for j in range(i + 1, lenLk_1):
            Li = list(Lk_1[i])[:k - 2]  # Lk_1[i]的 0 到 k-2 项
            Lj = list(Lk_1[j])[:k - 2]  # Lk_1[j]的 0 到 k-2 项
            Li.sort()
            Lj.sort()
            if Li == Lj:
                Ck.append(Lk_1[i] | Lk_1[j])
    return Ck


def apriori(dataSet, minSupport=0.5):
    """
    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return: 所有阶频繁项集合，支持度字典
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]  # 包含L1 L2 ... Li
    k = 2
    while (len(L[k - 2]) > 0):  # 当 Li 为空集（即不存在更高阶频繁项）时，循环停止
        # Ck 代表 k 阶候选项集合
        # Lk 代表 k 阶频繁项集合(Ck 经过筛选后，符合条件的子集，也就是 C k+1)
        Ck = aprioriGetCk(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    """
    :param L:
    :param supportData:
    :param minConf: 最小可信阈值（玄学参数）
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  # 候选规则右部的集合
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, bigRuleList, minConf=0.7):
    """
    :param freqSet:
    :param H:
    :param supportData: 存储了各阶候选项，频繁项的支持度
    :param bigRuleList:
    :param minConf:
    :return: 该轮通过检测的规则
    """
    prunedH = []
    for conseq in H:
        # 规则 P -> H 的可信度定义为：support[P | H] / support[P]
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算可信度
        if conf >= minConf:
            bigRuleList.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# 生成更多规则
def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGetCk(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, bigRuleList, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)


'''
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    # 将原始数据转换为list，python2 与 python3有区别：
    # python2 D = map(set, dataSet)
    # python3 D = list(map(set, dataSet))
    D = list(map(set, dataSet))
    L1, supportData0 = scanD(D, C1, 0.5)
    print("L1=%s, supportData=%s" % (L1, supportData0))
'''

dataSet = loadDataSet()
C1 = createC1(dataSet)
L, supportData = apriori(dataSet)
rules = generateRules(L, supportData, minConf=0.5)

print("rules=%s" % rules)
