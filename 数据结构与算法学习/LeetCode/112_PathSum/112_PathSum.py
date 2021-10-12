# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:49:06 2018

@author: XPS13
"""
# 路径和，始终传递对于下一个结点的期望和，与结点的val对比
class Solution(object):
    def hasPathSum(self, root, sumObj):
        if root == None:
            return False
        if root.left == None and root.right == None:
            return root.val == sumObj
        
        leftResult = self.hasPathSum(root.left, sumObj-root.val)
        rightResult = self.hasPathSum(root.right, sumObj-root.val)
        return leftResult or rightResult