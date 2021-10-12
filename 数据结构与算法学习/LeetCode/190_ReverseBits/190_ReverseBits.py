# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:24:25 2019

@author: XPS13
"""

class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        ret, count = 0, 32
        for i in range(count):
            ret = (ret << 1) + (n & 1)
            n = n >> 1
        return ret