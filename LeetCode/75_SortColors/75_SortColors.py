# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:24:35 2019

@author: XPS13
"""
# Solution 1: Count sort, O(#colors) space, O(n) time
class Solution(object):
    def sortColors(self, nums):
        if len(nums) == 0:
            return []
        elif len(nums) == 1:
            return nums
        
        count = {0:0, 1:0, 2:0}
        for i in nums:
            count[i] += 1
        
        start = 0
        for i in count.keys():
            for j in range(count[i]):
                nums[start + j] = i
            start += count[i]
            
        return nums
            