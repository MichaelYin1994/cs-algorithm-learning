# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:44:11 2018

@author: XPS13
"""
# 字典不停的找对应的值，键是与target差的值，值是序号
# HashTable时间复杂度为O(1)。

class Solution(object):
    def twoSum(self, nums, target):
        if len(nums) == 0:
            return None
        
        resDict = {}
        for ind, item in enumerate(nums):
            if item in resDict:
                return [resDict[item], ind]
            else:
                resDict[target - item] = ind
        return None

if __name__ == "__main__":
    a = [9, 1, 7, 3, 12, 4, 51, 34]      
    s = Solution()
    ret = s.twoSum(a, 13)
        