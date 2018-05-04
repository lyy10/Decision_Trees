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

#采用递归算法训练决策树
#def ID3TreeGenerate(D, A):



def computInforEntropy(D,lable):
    """
        lable 需是数据的属性列表
        D 是训练或部分训练数据集，组织方式是二维数组, 这里使用 numpy
    """
    num_lable = len(lable)
    i = np.zeros(num_lable)
    # 计算每类样本所占的数量
    for item in D:
        for j in range(0,num_lable):
            if item[num_lable] == lable[j]:
                i[j] += 1
                break
    num = D.shape
    Ent_D = 0
    # 计算信息熵 
    for n in i:
        Ent_D += float(n)/num[0] * math.log(2, float(n)/num[0])
    return -Ent_D


def computInforGain(D, lable, item_number, item_value):
    """
        D 为数据集
        lable 为属性列表
        item_number 为属性位置编号
        item_value 为对应属性可能的取值，离散性，是一个列表
    """
    infor_entropy = computInforEntropy(D,lable)

