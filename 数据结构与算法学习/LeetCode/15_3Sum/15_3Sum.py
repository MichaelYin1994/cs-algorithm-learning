# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:10:40 2019

@author: XPS13
"""
class Solution:
    # @return a list of lists of length 3, [[val1,val2,val3]]
    def threeSum(self, nums):
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            # i大于0时，再判别nums数组递增的情况。
            if i > 0 and nums[i] <= nums[i-1]:
                continue
            else:
                left = i + 1
                right = len(nums) - 1
                while(left < right):
                    if nums[left] + nums[right] == -nums[i]:
                        res.append([nums[i], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]: left += 1
                        while left < right and nums[right] == nums[right + 1]: right -= 1
                    elif nums[left] + nums[right] < -nums[i]:
                        left += 1
                    else:
                        right -= 1
        return res

if __name__ == "__main__":
    test = sorted([0, 0, 0])
    s = Solution()
    res = s.threeSum(test)