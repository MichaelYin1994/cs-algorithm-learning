# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 03:25:35 2019

@author: XPS13
"""

class Solution(object):
    def lengthOfLIS(self, nums):
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return 1
        
        
        dp = [1] * len(nums)
        for ind, item in enumerate(nums[1:]):
            ind = ind + 1
            for j in range(0, ind):
                if item > nums[j]:
                    dp[ind] = max(dp[ind], dp[j] + 1)
        return max(dp)