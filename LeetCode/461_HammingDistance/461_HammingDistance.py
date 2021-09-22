# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:49:22 2019

@author: XPS13
"""

class Solution(object):
    def hammingDistance(self, x, y):
        count = 0
        while(x or y):
            res_x = x & 1
            res_y = y & 1
            if res_x != res_y:
                count += 1
            x = x >> 1
            y = y >> 1
        return count