# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:12:39 2019

@author: XPS13
"""
###############################################################################
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
###############################################################################
# 将排序数组转换成二叉搜索树(Height-Balanced树)，并返回头结点，只要像二分查找一样递归的
# 返回头结点就可以。
        
# 递归
class Solution(object):
    def sortedArrayToBST(self, nums):
        if len(nums) == 0:
            return None
        
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
    
# 迭代
class Solution_1:
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.build(nums, 0, len(nums)-1)
    
    def build(self, nums, low, high):
        if low == high:
            return TreeNode(nums[low])
        elif low < high:
            mid = (low+high)//2
            node = TreeNode(nums[mid])
            node.left = self.build(nums, low, mid-1)
            node.right = self.build(nums, mid+1, high)
            return node