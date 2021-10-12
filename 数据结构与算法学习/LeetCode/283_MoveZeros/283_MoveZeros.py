# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 21:31:23 2019

@author: XPS13
"""

class Solution:
    def moveZeroes(self, nums):
        if len(nums) == 0:
            return []
        
        zeroFlag = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zeroFlag] = nums[zeroFlag], nums[i]
                zeroFlag += 1
            