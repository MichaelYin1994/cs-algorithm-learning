# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:34:20 2019

@author: XPS13
"""
# 注意[0, 0, 2], [2, 0, 0, 1]等，检查最优性条件
class Solution(object):
    def canJump(self, nums):
        if len(nums) == 0:
            return False
        elif len(nums) == 1:
            return True
        
        currMaxRange = 0
        for i in range(len(nums) - 1):
            if i <= currMaxRange:
                currMaxRange = max(i + nums[i], currMaxRange)
            else:
                return False                                 # 说明没办法再往前跳 
        if currMaxRange >= len(nums) - 1:
            return True
        else:
            return False