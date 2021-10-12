# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:50:46 2019

@author: XPS13
"""

class Solution(object):
    def titleToNumber(self, s):
        if len(s) == 0:
            return 0
        count, ret = len(s) - 1, 0
        for c in s:
            ret += (ord(c) - 64) * (26 ** count)
            count -= 1
        return ret