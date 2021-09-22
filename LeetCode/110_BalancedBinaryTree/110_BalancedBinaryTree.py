# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 02:36:45 2018

@author: XPS13
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 注意这里需要递归的判断左右子树是不是深度之间相差1，但是若是不满足平衡二叉树的条件的
# 时候，应当返回标志位-1,。
class Solution(object):
    def check(self, node):
        if node == None:
            return 0
        leftDepth = self.check(node.left)
        rightDepth = self.check(node.right)
        
        # 此处判断标志位条件是不是被满足，若是满足不计算节点深度，直接返回标志位
        if leftDepth == -1 or rightDepth == -1 or abs(leftDepth - rightDepth)>1:
            return -1
        return 1 + max(leftDepth, rightDepth)
    
    def isBalanced(self, root):
        return self.check(root) != -1
    