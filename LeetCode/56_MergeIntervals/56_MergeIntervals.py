# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:34:47 2019

@author: XPS13
"""

class Solution(object):
    def merge(self, intervals):
        if len(intervals) == 0 or len(intervals) == 1:
            return intervals
        # Sorted the intervals by their start points
        intervals = sorted(intervals)
        
        # Using stack to process the intervals
        stack = [[intervals[0][0], intervals[0][1]]]
        for nums in intervals[1:]:
            if nums[0] > stack[-1][1]:
                stack.append(nums)
            elif nums[1] <= stack[-1][1]:
                continue
            else:
                stack[-1][1] = nums[1]
        return stack

if __name__ == "__main__":
    intervals = [[1, 3], [2, 4], [8, 12]]
    s = Solution()
    res = s.merge(intervals)