# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:35:54 2018

@author: XPS13
"""

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
# 详情见插图，preorder的第一个节点一定是根节点，在inorder中，找到该结点的
# 位置。该节点左边的一定是左子树，右边一定是右子树。
class Solution:
    def buildTree(self, preorder, inorder):
        if inorder:
            head = TreeNode(preorder.pop(0))
            loc = inorder.index(head.val)
            # 传递数组的切片到下一级，返回的是head节点的地址
            head.left = self.buildTree(preorder, inorder[:loc])
            head.right = self.buildTree(preorder, inorder[(loc+1):])
            return head