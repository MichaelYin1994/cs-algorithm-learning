# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:46:40 2019

@author: XPS13
"""
class Solution:
    def n_sum(self, nums, target, n, result, results):
        # 递归终止条件，数组数量不够或者是n小于2
        if len(nums) < n or n < 2:
            return []
        
        # 降低N-sum问题为2-sum问题
        if n == 2:
            left, right = 0, len(nums) - 1
            while left < right:
                if nums[left] + nums[right] == target:
                    results.append(result + [nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while right > left and nums[right] == nums[right + 1]:
                        right -= 1
                elif nums[left] + nums[right] < target:
                    left += 1
                else:
                    right -= 1
        else:
            # 遍历数组每一个元素，降低问题的难度
            for i in range(len(nums) - n + 1): # 谨慎的防止数组溢出
                # 缩减搜索空间，若是nums[0]*n与nums[-1]*n与target相比结果为False
                # 后面也没有搜索的必要了，直接break
                if target < nums[0] * n or target > nums[0] * n:
                    break
                if i == 0 or (i > 0 and nums[i-1] < nums[i]):
                    self.n_sum(nums[(i+1):], target-nums[i], n - 1, result+[nums[i]], results)
        return results
    
if __name__ == "__main__":
    nums = [1, 0, -1, 0, -2, 2]
    nums.sort()
    
    target = 0
    s = Solution()
    res = s.n_sum(nums, target, n=4, result=[], results=[])