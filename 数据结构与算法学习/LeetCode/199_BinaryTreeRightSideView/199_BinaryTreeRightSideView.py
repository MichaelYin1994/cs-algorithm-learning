# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 20:44:03 2018

@author: XPS13
"""

# 树节点的定义：
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 从最右边看，最右边的结点应该从BFS的角度来看，左边看的元素位于BFS后的数组的最
# 后一个元素。
class Solution(object):
    def rightSideView(self, root):
        if root == None:
            return []
        levelVal, currLevel = [], [root]
        ret = []
        while(currLevel):
            levelVal.extend([node.val for node in currLevel])
            ret.append(levelVal[-1])
            nextLevel = []
            for node in currLevel:
                nextLevel.extend([node.left, node.right])
            currLevel = [node for node in nextLevel if node != None]
        return ret