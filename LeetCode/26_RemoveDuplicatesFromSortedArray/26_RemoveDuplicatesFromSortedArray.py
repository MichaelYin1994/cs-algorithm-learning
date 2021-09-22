# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 08:35:38 2019

@author: XPS13
"""

class Solution:
    def removeDuplicates(self, nums):
        if len(nums) == 0:
            return 0
        
        count = 1
        front = rear = 0
        pos = 1
        arraySize = len(nums)
        
        while(rear <= arraySize - 1):
            if nums[front] == nums[rear]:
                rear += 1
            else:
                count += 1
                front = rear
                nums[pos] = nums[rear]
                pos += 1
        return count

if __name__ == "__main__":
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    s = Solution()
    res = s.removeDuplicates(nums)