# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:04:10 2018

@author: XPS13
"""
# 思路和62题一致，自下向上的Dynamic Programming，注意检测的是边界条件便可以
# 容易出错的检测案例：
# (1) [[0, 1, 0]]
# (2) [[0], [1], [0]]
# (3) [[1, 0], [0, 0]]
# O(mn)时间复杂度，若是在obstacleGrid上构建DP矩阵，空间复杂度为O(1)
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        self.col = len(obstacleGrid[0])
        self.row = len(obstacleGrid)
        # 检测边界条件(1), (2)
        if self.col == 1:
            for i in range(len(obstacleGrid)):
                if obstacleGrid[i][0] == 1:
                    return 0
                else:
                    continue
            return 1
        elif self.row == 1:
            for i in range(len(obstacleGrid[0])):
                if obstacleGrid[0][i] == 1:
                    return 0
                else:
                    continue
            return 1
        
        tmp = [ 0 ] * (self.col + 1)
        memory = []
        
        # 检测边界条件(3)
        if obstacleGrid[0][0] == 0:
            memory[1][1] = 1
        else:
            memory[1][1] = 0
        
        for i in range(self.row + 1):
            memory.append(tmp)
            
        # 构建DP矩阵
        for i in range(1, self.row + 1):
            for j in range(1, self.col + 1):
                if i == 1 and j == 1:
                    continue
                elif obstacleGrid[i-1][j-1] != 1:
                    memory[i][j] = memory[i-1][j] + memory[i][j-1]
                else:
                    memory[i][j] = 0
        return memory[-1][-1]