# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:16:27 2019

@author: XPS13
"""
# 双指针跑过去就行，注意单递增序列，数字全相同序列
class Solution(object):
    def findLengthOfLCIS(self, nums):
        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return 1
        
        slow, fast = 0, 1 
        maxLength, currMaxLength = 1, 0 # maxLength始终大于等于1，不如直接赋1
        while(fast <= len(nums) - 1):
            if nums[fast] > nums[fast - 1]:
                currMaxLength = fast - slow + 1
            else:
                slow = fast
                maxLength = currMaxLength if currMaxLength > maxLength else maxLength
            fast += 1
            
        # 为了防止单递增序列，else语句不执行
        return max(maxLength, currMaxLength)
            
if __name__ == "__main__":
    nums = [1, 3, 5, 7]
    nums = [2, 2, 2, 2, 2]
    s = Solution()
    res = s.findLengthOfLCIS(nums)