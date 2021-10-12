# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:17:58 2018

@author: XPS13
"""
# 这种题，使用带环链表的思想来套。首先确定，第一个元素一定可以作为链表的入口，不是自环（因为
# 规定数字范围从1 -- n，第一个元素下标是0）；其次可以分两种情况考虑：
# 第一种是有入口结点，然后入环；
# 第二种是从入口直接进入了环。
# 综合考虑，这道题的快指针和慢指针一定要从同一个地方开始，否则就有进入自环的可能（例如[4, 1, 2, 3, 3]）

class Solution(object):
    def findDuplicate(self, nums):
        if len(nums) == 2:
            return nums[0]
        
        pSlow = nums[0]
        pFast = nums[0]
        while(True):
            pSlow = nums[pSlow]
            pFast = nums[nums[pFast]]
            if pSlow == pFast:
                break
        
        pSlow = nums[0]
        while(True):
            # 首先判断是不是pFast和pSlow在同一位置，防止自环的死循环
            if pSlow == pFast:
                return pSlow
            pFast = nums[pFast]
            pSlow = nums[pSlow]
