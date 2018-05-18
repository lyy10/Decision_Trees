#
# ID3 program test
# May,18,2018 @ Lyy



def TestRight(data,decisiontree,A):
    if decisiontree.isleaf == 1:
        if data[-1] == decisiontree.item_lable:
            return 1
        else:
            return 0

    temp = decisiontree.item_value[str(int(data[A.item_name.index(decisiontree.item_name)]))]
    return TestRight(data,temp,A)
def Test(dataSet,head,A):
    first_node = head.item_value['head']
    num_right = 0
    for data in dataSet:
        if TestRight(data,first_node,A):
            num_right += 1
    print("right num :" + str(num_right))
    print(num_right/len(dataSet))
