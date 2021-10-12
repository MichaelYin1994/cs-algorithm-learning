# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:46:14 2018

@author: XPS13
"""
# 合并两个排序数组，注意和归排的相似性。并且从最大的值开始合并，而不是从最小的值开始。
class Solution:
    def merge(self, nums1, m, nums2, n):
        while( m > 0 and n > 0):
            # 找出nums1的最后一个元素和nums2的最后一个元素哪一个大，然后添加到
            # nums1的最末尾去
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        # 若是nums1元素先用完，那么nums2必然还剩下n个元素没有用，而len(nums1) = m + n
        # 所以nums1还剩n个没能排序
        if n > 0:
            nums1[:n] = nums2[:n]


class Solution_my(object):
    def merge(self, nums1, m, nums2, n):
        if n == 0:
            return
        
        pointer_1 = m - 1
        pointer_2 = n - 1
        pos = m + n - 1
        
        while(pointer_1 >= 0 and pointer_2 >= 0):
            if nums1[pointer_1] > nums2[pointer_2]:
                nums1[pos] = nums1[pointer_1]
                pos -= 1
                pointer_1 -= 1
            elif nums1[pointer_1] <= nums2[pointer_2]:
                nums1[pos] = nums2[pointer_2]
                pos -= 1
                pointer_2 -= 1
        
        if pointer_1 < 0:
            while(pos >= 0):
                nums1[pos] = nums2[pointer_2]
                pos -= 1
                pointer_2 -= 1