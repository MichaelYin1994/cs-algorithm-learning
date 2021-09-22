# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 02:08:00 2018

@author: XPS13
"""
# 标准DP问题，不能用DFS
class Solution(object):
    def uniquePaths(self, m, n):
        if m == 1 or n == 1:
            return 1
        self.col = m
        self.row = n
        tmp = [ 0 ] * (self.col + 1)
        memory = []
        for i in range(self.row + 1):
            memory.append(tmp)
        
        memory[1][1] = 1
        for i in range(1, self.row + 1):
            for j in range(1, self.col + 1):
                if i == 1 and j == 1:
                    continue
                else:
                    memory[i][j] = memory[i-1][j] + memory[i][j-1] + memory[i-1][j-1]
        return memory[-1][-1]

if __name__ == "__main__":
    s = Solution()
    ret = s.uniquePaths(50, 50)