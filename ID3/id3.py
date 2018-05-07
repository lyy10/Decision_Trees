# -*- coding: utf-8 -*-
#
#ID3 decision trees 
#use Gain() 信息增益函数选择最优划分属性
# 2018-05-04 @ Lyy

import numpy as np
import math
#定义决策树节点
class ID3_Node(object):
    def __init__(self):
        self.item_name = 'null'     #属性名
        self.item_value = {}        #属性值，是一个字典，每种取值对应一个节点
        self.isleaf = 0             #节点标志位，默认 0 为非叶节点
        self.item_lable = 'null'    #当节点标志位为 1 时，表示该节点的类别

#配置类，针对不同数据集经行人工初始化
class Config(object):
    def __init__(self):
        # 属性名列表
        self.item_name = []
        # 属性向量，初始值为1
        self.item = []
        # 每个属性名，对应的属性可能取值列表，所以为二维数组
        self.item_value = []
        # 标签名列表
        self.lable = []

#采用递归算法训练决策树
#def ID3TreeGenerate(D, A):



def computInforEntropy(D,lable):
    """
        lable 需是数据的标签列表
        D 是训练或部分训练数据集，组织方式是二维数组, 这里使用 numpy
    """
    num_lable = len(lable)
    i = np.zeros(num_lable)
    # 计算每类样本所占的数量
    for item in D:
        for j in range(0,num_lable):
            if item[-1] == lable[j]:
                i[j] += 1
                break
    num = D.shape
    Ent_D = 0
    # 计算信息熵 
    for n in i:
        if n != 0:
            Ent_D += float(n)/num[0] * math.log(float(n)/num[0], 2)
    return -Ent_D


def computInforGain(D, lable, item_number, item_value):
    """
        D 为数据集
        lable 为标签列表
        item_number 为属性位置编号
        item_value 为对应属性可能的取值，离散性，是一个列表
    """
    num_D = D.shape
    infor_entropy = computInforEntropy(D,lable)
    temp = []
    for i in item_value:
        temp.append([])
    for item in D:
        for i in range(0,len(item_value)):
            if item[item_number] == item_value[i]:
                temp[i].append(item)
                break
    each_sub = 0
    for i in range(0,len(item_value)):
        if temp[i]:
            each_sub += float(len(temp[i]))/num_D[0] * computInforEntropy(np.array(temp[i]),lable)
    return infor_entropy - each_sub


def selectItem(D, item, item_value, lable, conse_item = []):
    """
        选择最优划分属性
        D 是数据集
        item 属性列表
        conse_item 是连续属性列表，默认为空
    """
    Gain = []
    for i in range(0,len(item)):
        if item[i] != 0:
            Gain.append(computInforGain(D,lable,i,item_value[i]))
        else:
            Gain.append(-1)
    index = Gain.index(max(Gain))
    return index


def ID3TreeGenerate(D, A, nextnode, action):
    """
        A 是位置类
        nextnode 是传入的节点
        action 分支类，比如当属性
    """
    node = ID3_Node()
    nextnode.item_value[action] = node
    num = np.zeros(len(A.lable))
    #找出每一类的样本数量
    for i in range(0,D.shape[0]):
        for j in range(0,len(A.lable)):
            if D[i][-1] == A.lable[j]:
                num[j] += 1
                break
    #处理当属性为空，样本集属于同一类，每个样本属性值都相同的情况
    #置为叶节点
    if max(A.item)==0 | num.max() == D.shape[0]:
        node.isleaf = 1
        node.item_lable = A.lable[num.argmax()]
        return
    temp = D[0]
    sign = 1
    for item in D:
        if True in temp != item:
            sign = 0
            break
    if sign:
        node.isleaf = 1
        node.item_lable = A.lable[num.argmax()]
        return
    #选择最优属性
    opt_index = selectItem(D,A.item,A.item_value,A.lable)
    A.item[opt_index] = 0
    Dv = []
    #对于属性的每个取值，划分出对应数据集
    for item in A.item_value[opt_index]:
        Dv.append(D[D[:,opt_index]==item,:])
    for i in range(0,len(A.item_value)):
        
        if Dv[i]:
            ID3TreeGenerate(Dv[i],A,node,str(A.item_value[i]))
        else:
            tempnode = ID3_Node()
            node.item_value[str(A.item_value[i])] = tempnode
            tempnode.isleaf = 1
            tempnode.item_lable = A.lable[num.argmax()]
def StartTrain():
    '''
        训练决策树起始函数
    '''
    # 由数据文本或者数据库生成总训练集表，当然本程序更适用于文本数据
    D = []
    A = Config()
    # init


    head = ID3_Node()
    head.item_name = 'head'
    head.item_value['head'] = []
    ID3TreeGenerate(D,A,head,'head')
