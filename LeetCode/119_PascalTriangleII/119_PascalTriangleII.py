# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:26:55 2019

@author: XPS13
"""
class Solution(object):
    def getRow(self, rowIndex):
        if rowIndex == 0:
            return [1]
        elif rowIndex == 1:
            return [1, 1]
        elif rowIndex == 2:
            return [1, 2, 1]
        
        row = 2
        dp = [1, 2, 1]
        while(row + 1 <= rowIndex):
            tmp = [1]
            for i in range(1, len(dp)):
                tmp.append(dp[i] + dp[i-1])
            tmp.append(1)
            dp = tmp
            row += 1
        return dp
