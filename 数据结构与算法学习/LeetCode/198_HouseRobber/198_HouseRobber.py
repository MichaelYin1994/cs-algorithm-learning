# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:51:28 2019

@author: XPS13
"""
# dp = [0, 0]
# f(n) = max(dp[n-2] + nums[n], dp[n-1])
class Solution:
    def rob(self, nums):
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums)
        
        dp = [0, 0]
        for i in range(len(nums)):
            dp.append(max(dp[i+2-2] + nums[i], dp[i+2-1]))
        return dp[-1]