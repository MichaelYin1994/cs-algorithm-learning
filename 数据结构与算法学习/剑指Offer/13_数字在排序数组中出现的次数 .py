# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:34:07 2018

@author: XPS13
"""
# 特别注意上界与下界的关系
class Solution:
    def binary_search(self, low, high, target):
        while(low <= high):
            mid = low + (high - low)//2
            if self._array[mid] > target:
                high = mid + 1
            elif self._array[mid] < target:
                low = mid - 1
            elif self._array[mid] == target:
                return mid
        return None
    
    # 查找下界，因为mid（上界）的值已经可以等于target，而上界还在降低
    # 所以是找到的下界。
    def lower_bound(self, low, high, target):
        while(low <= high):
            mid = low + (high - low)//2
            if self._array[mid] >= target:
                high = mid - 1
            elif self._array[mid] < target:
                low = mid + 1
        return low
    
    # 查找上界，因为mid（下界）的值已经可以等于target，而下界还在升高
    # 所以是找到的上界。但是由于最后一步执行了mid + 1，所以返回的时候
    # 需要减去1，才是元素真正的上界。
    def upper_bound(self, low, high, target):
        while(low <= high):
            mid = low + (high - low)//2
            if self._array[mid] > target:
                high = mid - 1
            elif self._array[mid] <= target:
                low = mid + 1
        return low - 1
    
    def GetNumberOfK(self, data, k):
        if len(data) == 0:
            return 0
        self._array = data
        low = 0
        high = len(data) - 1
        # 先检查数字存在不存在，然后再查找上下界。
        if self.binary_search == None:
            return None
        else:
            upper_bound = self.upper_bound(low, high, k)
            lower_bound = self.lower_bound(low, high, k)
            counter = upper_bound - lower_bound + 1
        return counter