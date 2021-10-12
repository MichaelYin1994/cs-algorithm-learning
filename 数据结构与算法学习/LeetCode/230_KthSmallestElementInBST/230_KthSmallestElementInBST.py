# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:48:34 2018

@author: XPS13
"""
# 使用中序遍历，遍历树的所有结点，并且选出第k个最小的树就可以了
class Solution:
    def kthSmallest(self, root, k):
        if root == None:
            return None
        ret = []
        stack = []
        # 迭代终止条件是堆栈空和结点空都成立
        while (stack or root):
            if root:
                stack.append(root)
                root = root.left
            else:
                tmp = stack.pop()
                ret.append(tmp.val)
                k -= 1
                root = tmp.right
            if k == 0:
                break
        return ret[-1]