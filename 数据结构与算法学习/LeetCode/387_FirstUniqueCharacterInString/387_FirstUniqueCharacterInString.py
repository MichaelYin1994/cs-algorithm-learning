# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 22:30:03 2018

@author: XPS13
"""
# 自己的解法是强行解。更优的解法：字典加队列。字典保持字符的个数，队列保持字符第一次
# 出现的位置。
class Solution(object):
    def firstUniqChar(self, s):
        table = {}
        for ind, c in enumerate(s):
            if c in table.keys():
                table[c][1] = True
            else:
                table[c] = [ind, False]
        
        maxIndex = len(s) + 1
        for c in table.keys():
            if table[c][1] == False:
                maxIndex = table[c][0] if table[c][0] < maxIndex else maxIndex
        return maxIndex if maxIndex != len(s) + 1 else -1

