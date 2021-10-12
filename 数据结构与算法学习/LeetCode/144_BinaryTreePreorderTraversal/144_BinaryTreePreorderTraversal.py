# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 01:04:17 2018

@author: XPS13
"""
# 思路和中序遍历非常相似，参考中序遍历
class Solution(object):
    def __init__(self):
        Solution.ans = []
    def preorderTraversal(self, root):
        if root != None:
            Solution.ans.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)
        return Solution.ans

class SolutionIteratively(object):
    def preorderTraversal(self, root):
        ans = []
        stack = []
        while(stack or root):
            if root != None:
                ans.append(root.val)
                stack.append(root)
                root = root.left
            else:
                tmpNode = stack.pop()
                root = tmpNode.right
        return ans