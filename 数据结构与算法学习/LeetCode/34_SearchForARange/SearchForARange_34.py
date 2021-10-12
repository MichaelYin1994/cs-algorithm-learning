# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:50:48 2018

@author: Administrator
"""
class Solution:
    def lower_bound(self, nums, target):
        low = 0
        high = len(nums)-1
        while(low <= high):
            mid = low + (high - low) // 2
            if (nums[mid] >= target):
                high = mid - 1
            elif (nums[mid] < target):
                low = mid + 1
        return low

    def upper_bound(self, nums, target):
        low = 0
        high = len(nums) - 1
        while(low <= high):
            mid = low + (high - low) // 2
            if (nums[mid] > target):
                high = mid - 1
            elif (nums[mid] <= target):
                low = mid + 1
        return low-1
    
    def searchRange(self, nums, target):
        if len(nums)==0:
            return [-1, -1]
        elif (target < nums[0] or target > nums[-1]):
            return [-1, -1]
        
        lowInd = self.lower_bound(nums, target)
        if target == nums[lowInd]:
            upInd = self.upper_bound(nums, target)
            return [lowInd, upInd]
        else:
            return [-1, -1]
        
if __name__ == "__main__":
    nums = [1, 2, 3, 4]
    target = 2
    s = Solution()
    res = s.searchRange(nums=nums, target=target)