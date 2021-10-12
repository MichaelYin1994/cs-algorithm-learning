# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:40:09 2018

@author: XPS13
"""
# 二叉树的广度优先搜索BFS，关键点在于建立两个列表
# 第一个用来保存值，第二个用来保存该层所有的结点的地址
class Solution:
    def levelOrder(self, root):
        if root == None:
            return []
        ret, currLevel = [], [root]
        while currLevel:
            ret.append([node.val for node in currLevel])
            nextLevel = []
            # 拓展下一层列表，保存下一层的地址
            for node in currLevel:
                nextLevel.extend([node.left, node.right])
            # 将currLevel的结点地址给构造出来，None结点在这层就被排除掉了
            currLevel = [node for node in nextLevel if node != None]
        return ret