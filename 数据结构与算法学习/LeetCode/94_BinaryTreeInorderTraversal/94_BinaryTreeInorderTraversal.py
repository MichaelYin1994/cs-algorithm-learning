# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 00:34:33 2018

@author: XPS13
"""
# 树的非递归中序遍历，关键点在于维护一个堆栈，堆栈要时刻保存上一个节点的地址
# 递归版本很简单
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def __init__(self):
        Solution.ans = []   
    def inorderTraversal(self, root):
        if root != None:
            self.inorderTraversal(root.left)
            Solution.ans.append(root.val)
            self.inorderTraversal(root.right)
        return Solution.ans

class SolutionIteratively(object):
    def inorderTraversal(self, root):
        # 维持一个堆栈，堆栈不停的存储上一级节点的地址，直到下一级为
        # None，就把节点pop出来。pop出来的节点是父亲节点，在继续将root
        # 赋值右节点的地址
        ans = []
        stack = []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                tmpNode = stack.pop()
                ans.append(tmpNode.val)
                root = tmpNode.right
        return ans