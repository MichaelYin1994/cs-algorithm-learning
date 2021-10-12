# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 00:41:29 2018

@author: XPS13
"""
# 关键点在于：总是左结点的左孩子的值与右结点的右孩子的值要相等才是对称的二叉树
# 若是两个结点的孩子都是None，没问题，可以返回True；若是有其中一方不是None，就要
# 返回False。
class Solution:
    # isMirror()函数用来判断左右孩子的值是不是对称，return的值判断左右孩子的左右孩子
    # 的值是不是对称，所以是递归形式，都返回True才是对称。
    def isMirror(self, nodeLeft, nodeRight):
        if (nodeLeft is None and nodeRight is None):
            return True
        elif (nodeLeft is None or nodeRight is None):
            return False
        return (nodeLeft.val == nodeRight.val) and (self.isMirror(nodeLeft.left, nodeRight.right)) and (self.isMirror(nodeLeft.right, nodeRight.left))
    
    def isSymmetric(self, root):
        if root == None:
            return True
        return self.isMirror(root.left, root.right)
            
