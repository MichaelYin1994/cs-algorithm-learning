# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:10:33 2018

@author: XPS13
"""
# 基本思路还是DP的思路
class Solution:
    # Top down
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        if len(triangle) == 1:
            return triangle[0][0]
        elif len(triangle) == 0:
            return None

        level = len(triangle)
        for i in range(1, level):
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i-1][j]
                elif j < (len(triangle[i]) - 1):
                    triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
                elif j == (len(triangle[i]) - 1):
                    triangle[i][j] += triangle[i-1][-1]
        return min(triangle[-1])