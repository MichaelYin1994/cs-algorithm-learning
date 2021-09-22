# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:33:56 2019

@author: XPS13
"""

class Solution(object):
    def increasingTriplet(self, nums):
        if len(nums) <= 1:
            return False
        
        first = second = float("inf")
        for n in nums:
            if n <= first:
                first = n
            elif n <= second:
                second = n
            else:
                return True
        return False