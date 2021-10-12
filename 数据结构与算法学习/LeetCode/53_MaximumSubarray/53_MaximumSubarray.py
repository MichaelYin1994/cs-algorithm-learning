# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:27:38 2019

@author: XPS13
"""

class Solution(object):
    def maxSubArray(self, nums):
        res = before = nums[0]
        for num in nums[1:]:
            if before >= 0:
                before = before + num
            else:
                before = num
            if res < before:
                res = before
        return res

if __name__ == "__main__":
    array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    s = Solution()
    print(s.maxSubArray(array))