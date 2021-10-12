# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:54:21 2018

@author: XPS13
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
# 思路相似于105， 后续遍历的最后一个结点一定是根节点，倒数第二个结点一定是根结点的
# 下一个结点，所以一直从右子树开始重建就好了。
class Solution:
    def buildTree(self, inorder, postorder):
        if inorder:
            root = TreeNode(postorder[-1])
            del postorder[-1]
            pos = inorder.index(root.val)
            root.right = self.buildTree(inorder[(pos+1):], postorder)
            root.left = self.buildTree(inorder[:pos], postorder)
            return root