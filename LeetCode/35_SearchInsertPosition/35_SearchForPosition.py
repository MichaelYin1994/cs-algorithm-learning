# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:20:02 2018

@author: Administrator
"""

class Solution:
    def binarySearch(self, nums, target):
        low = 0
        high = len(nums)-1
        
        while(low <= high):
            mid = low + (high - low) // 2
            if (nums[mid] > target):
                high = mid - 1
            elif (nums[mid] < target):
                low = mid + 1
            elif (nums[mid] == target):
                return mid
        return low
        
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        ind = self.binarySearch(nums, target)
        return ind

if __name__ == "__main__":
    nums = [1,3,5,7]
    target = 6
    s = Solution()
    res = s.searchInsert(nums, target)