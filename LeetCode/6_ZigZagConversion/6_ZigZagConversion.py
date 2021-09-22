# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 00:41:37 2018

@author: XPS13
"""

class Solution(object):
    def convert(self, s, numRows):
        if len(s) == 0:
            return ""
        ret = ""
        gap = numRows * 2 - 2
        index = [0]
        for ind in range(0, numRows):
            while(ind <= len(s)-1 and gap > 0):
                ret += s[ind]
                ind += gap
            gap -= 2
        return ret

string = "PAYPALISHIRING"
numRows = 4     
s = Solution()
res = s.convert(string, numRows)
solution = "PINALSIGYAHRPI"