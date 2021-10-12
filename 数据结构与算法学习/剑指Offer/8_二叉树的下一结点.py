# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # 先判断结点本身是不是None
        if pNode == None:
            return None
        # 若是右子树存在，中序遍历的下一结点一定是对
        # 该结点进行中序遍历
        if pNode.right != None:
            pNode = pNode.right
            while(pNode.left != None):
                pNode = pNode.left
            return pNode
        # 若是左子树存在，并且该结点的头结点也存在
        # 那么中序遍历的下一结点一定是该结点的头结点
        elif pNode.left != None and pNode.next != None:
            return pNode.next
        # 若是都不存在，则判断该节点是左节点还是右结点
        # 若是左节点并且上一结点存在，则中序遍历是上一结点
        # 若是右结点并且上一结点存在，则找到某一结点pNode
        # pNode.left == pNode的结点，该结点一定是中序遍历的
        # 下一结点
        elif pNode.left == None and pNode.right == None and pNode.next != None:
            if pNode.next.left == pNode:
                return pNode.next
            else:
                while(pNode.next != None and pNode.next.left != pNode):
                    pNode = pNode.next
                if pNode.next == None:
                    return None
                return pNode.next
        # 若是都不满足条件，则返回None，因为找到头也没有满足条件的结点。
        return None