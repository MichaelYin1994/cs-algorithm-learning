# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:13:23 2018

@author: XPS13
"""
# 剑指offer的一道难题，二叉搜索树的后序遍历序列，难点在于终止条件：不需要列举
# 递归终止条件，容易陷进去，转而判断数据的长度进行终止。

class Solution:
    def VerifySquenceOfBST(self, nums):
        if len(nums) == 0 or nums == None:
            return []
        root = nums[-1]
        elementNums = len(nums)
        i = 0
        while(i < elementNums - 1):
            if nums[i] < root:
                i += 1
            else:
                break
        j = i
        while(j < elementNums - 1):
            if nums[j] > root:
                j += 1
            else:
                return False
        # 判断左半数组的长度    
        leftResult = True
        if i > 0:
            leftResult = self.VerifySquenceOfBST(nums[:i])
        
        # 判断右半数组的长度
        rightResult = True
        if elementNums - i - 1 > 0:
            rightResult = self.VerifySquenceOfBST(nums[i:(elementNums-1)])
        return leftResult and rightResult