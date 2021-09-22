# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:01:29 2018

@author: XPS13
"""
# uniqueSet使用Python的集合类，set里的元素不允许重复，所以要熟悉set的方法
class Solution(object):
    def containsDuplicate(self, nums):
        if len(nums) == 0:
            return False
        uniqueSet = set()
        for i in nums:
            if i in uniqueSet:
                return True
            else:
                uniqueSet.add(i)
        return False