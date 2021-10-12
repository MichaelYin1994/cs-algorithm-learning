# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 00:06:25 2019

@author: XPS13
"""

#class Solution(object):
#    def in_order_traversal(self, currNumber, currRes, remainNums):
#        currRes.append(currNumber)
#        if len(remainNums) == 0:
#            self.ret.append(currRes)
#        else:
#            for ind, item in enumerate(remainNums):
#                if ind == len(remainNums) - 1:
#                    self.in_order_traversal(item, currRes, remainNums[:ind])
#                else:
#                    self.in_order_traversal(item, currRes, remainNums[:ind] + remainNums[(ind + 1):])
#
#    def permute(self, nums):
#        if len(nums) == 0:
#            return []
#        elif len(nums) == 1:
#            return nums
#        elif len(nums) == 2:
#            return [[nums[0], nums[1]], [nums[1], nums[0]]]
#        
#        self.ret = []
#        for ind, item in enumerate(nums):
#            if ind == len(nums) - 1:
#                self.in_order_traversal(item, [], nums[:ind])
#            else:
#                self.in_order_traversal(item, [], nums[:ind] + nums[(ind + 1):])
#        return self.ret
class Solution(object):
    def depth_first_search(self, res, path, remains):
        if len(remains) == 0:
            self.ret.append(path + [res])
        else:
            for ind, item in enumerate(remains):
                self.depth_first_search(item, path + [res], remains[:ind] + remains[(ind + 1):])
    def permute(self, nums):
        if len(nums) == 0:
            return []
        elif len(nums) == 1:
            return [nums]
        elif len(nums) == 2:
            return [[nums[0], nums[1]], [nums[1], nums[0]]]
        
        self.ret = []
        for ind, item in enumerate(nums):
            self.depth_first_search(item, [], nums[:ind] + nums[(ind + 1):])
        return self.ret


if __name__ == "__main__":
    nums = [1, 2, 3]
    s = Solution()
    ret = s.permute(nums)