# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:37:14 2018

@author: XPS13
"""
# 利用异或操作，还不太懂
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = 0
        for i in nums:
            result = result ^ i
        return result