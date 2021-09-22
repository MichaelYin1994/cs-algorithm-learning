# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 10:35:13 2018

@author: XPS13
"""
class Solution:
    def invertTree(self, root):
        if root == None:
            return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root