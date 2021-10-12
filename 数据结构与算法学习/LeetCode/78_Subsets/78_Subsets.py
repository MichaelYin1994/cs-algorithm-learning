# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 01:00:11 2019

@author: XPS13
"""
class Solution(object):
    def depth_first_search(self, res, path, remains):
        self.ret.append(path + [res])
        for ind, item in enumerate(remains):
            self.depth_first_search(item, path + [res], remains[(ind + 1):])

    def subsets(self, nums):
        if len(nums) == 0:
            return []
        elif len(nums) == 1:
            return [nums, []]
        
        self.ret = [[]]
        for ind, item in enumerate(nums):
            self.depth_first_search(item, [], nums[(ind + 1):])
        return self.ret

if __name__ == "__main__":
    nums = [1, 2, 3]
    s = Solution()
    ret = s.subsets(nums)