# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:26:50 2019

@author: XPS13
"""

class Solution(object):
    def generate(self, numRows):
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]
        elif numRows == 0:
            return []
        
        ind = 1
        triangle = [[1], [1, 1]]
        while(ind + 1 < numRows):
            res = [1]
            for i in range(1, len(triangle[-1])):
                res.append(triangle[-1][i] + triangle[-1][i-1])
            res.append(1)
            triangle.append(res)
            ind += 1
        return triangle