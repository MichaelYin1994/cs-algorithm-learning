# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:53:42 2019

@author: XPS13
"""

class SUM(object):
    def missingNumber(self, nums):
        return sum([i for i in range(len(nums) + 1)]) - sum(nums)
    
class XOR(object):
    def missingNumber(self, nums):
        # ind + 1是为了对齐下标
        res = 0
        for ind, item in enumerate(nums):
            res = res ^ (ind + 1)
            res = res ^ item
        return res