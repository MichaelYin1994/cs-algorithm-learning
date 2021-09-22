# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:17:51 2019

@author: XPS13
"""

# Solution 1：in-order遍历，中序遍历检查结果，顺便早停缩小搜索空间
class Solution(object):
    def isValidBST(self, root):
        if root == None:
            return True
        
        stack = [root]
        values = []
        node = root.left
        while(stack or node):
            if node != None:
                stack.append(node)
                node = node.left
            else:
                tmpNode = stack.pop()
                values.append(tmpNode.val)
                node = tmpNode.right
            if len(values) >= 2 and values[-1] <= values[-2]:
                return False
        return True

# Solution 2：修改入口，递归得解
class Solution_1(object):
    def isValidBST(self, root, lessThan = float('inf'), largerThan = float('-inf')):
        # 递归终止条件
        if not root:
            return True
        if root.val <= largerThan or root.val >= lessThan:
            return False
        return self.isValidBST(root.left, min(lessThan, root.val), largerThan) and \
               self.isValidBST(root.right, lessThan, max(root.val, largerThan))