# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 02:11:15 2018

@author: XPS13
"""
# 见Leetcode解释，不是非常的懂，但是思考过程是这样的：
# 1. 先从最左和最右开始找起，因为这两个隔板最宽。计算其容量(right - left) * min(height[left], height[right])。
# 2. 计算完最左和最右以后，将最小的高度对应的指针移动一下（容量取决于最短的那块，这样才可能提高其容量），不停的算容量。
class Solution(object):
    def maxArea(self, height):
        maxVolume = 0
        left = 0
        right = len(height) - 1
        while(left < right):
            volumeTmp = (right - left) * min(height[left], height[right])
            if volumeTmp > maxVolume:
                maxVolume = volumeTmp
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return maxVolume

a = [1,8,6,2,5,4,8,3,7]
s = Solution()
res = s.maxArea(a)
