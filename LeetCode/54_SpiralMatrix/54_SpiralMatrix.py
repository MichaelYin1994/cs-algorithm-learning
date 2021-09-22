# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:49:11 2018

@author: XPS13
"""
# 此题该解法最优，思路在于简历建立4个索引，分别代表行，行的边界，列，列的边界
class Solution:
    def spiralOrder(self, matrix):
        if len(matrix) == 0:
            return []
        ret = []
        m = len(matrix)
        n = len(matrix[0])
        a, b, c, d = 0, m-1, 0, n-1
        # 此处条件一定是小于，若是等于的话，坐标会再运行一步，导致a<b成立，导致while
        # 后面索引麻烦。
        while(a < b and c < d):
            ret.extend(matrix[a][i] for i in range(c, d)) 
            ret.extend(matrix[i][d] for i in range(a, b))
            ret.extend(matrix[b][i] for i in range(d, c, -1))
            ret.extend(matrix[i][c] for i in range(b, a, -1))
            a, b, c, d = a+1, b-1, c+1, d-1
        
        # 仅会有这两种情况出现
        if a == b:
            ret.extend(matrix[a][i] for i in range(c, d+1))
        elif c == d:
            ret.extend(matrix[i][c] for i in range(a, b+1))
        return ret