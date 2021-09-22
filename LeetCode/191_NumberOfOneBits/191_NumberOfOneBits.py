# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:54:09 2018

@author: XPS13
"""
# 不停的与0b00000000000000000001做位与运算
class Solution(object):
    def hammingWeight(self, n):
        count = 0
        while(n):
            if n & 1:
                count += 1
            n = n >> 1
        return count