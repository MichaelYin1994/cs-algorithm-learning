# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:26:07 2018

@author: XPS13
"""
# 错误答案，仍待思考验证
class Solution:
    def findMin(self, nums):
        if nums == None:
            return None
        low = 0
        high = len(nums) - 1
        while(low <= high):
            if nums[low] <= nums[high]:
                return nums[low]
            mid = low + (high - low) // 2
            if nums[mid] > nums[low]:
                low = mid + 1
            elif nums[mid] < nums[high]:
                high = mid

# 正确答案，若是nums[mid]>=nums[low]，说明low部分是顺序的，
# 那么让low = mid，可以减小搜索范围；
# 若是nums[mid] <= nums[high] 说明high部分是顺序的，让mid = high
# 也可以减小搜索范围；直到最后，二者的指针差值会只有1个，也就是high - low = 1
# 因为每次都是原地移动(high=mid, low=mid)。
# 当只差一个的时候，high一定是最高值。详见《剑指offer》P83

class Solution(object):
    def findMin(self, nums):
        if len(nums) == 0:
            return None
        
        # 提前判断顺序的情况，减小计算量
        if nums[0] <= nums[-1]:
            return nums[0]
        
        low = 0
        high = len(nums) - 1
        while(high - low != 1):
            mid = low + (high-low)//2
            if nums[low] <= nums[mid]:
                low = mid
            elif nums[high] >= nums[mid]:
                high = mid
        return nums[high]

#         return None
            
    # [4,5,6,7,0,1,2]
    # [4,5,6,0,1,2,3]
    # [0,1,2,3]
    
    
#      int left = 0,  right = nums.size() - 1;
#     while(left < right) {
#         if(nums[left] < nums[right]) 
#             return nums[left];
            
#         int mid = (left + right)/2;
#         if(nums[mid] > nums[right])
#             left = mid + 1;
#         else
#             right = mid;
#     }
    
#     return nums[left];