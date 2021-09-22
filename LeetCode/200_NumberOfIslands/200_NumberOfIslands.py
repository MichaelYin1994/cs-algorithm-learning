# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:35:36 2019

@author: XPS13
"""

class Solution(object):
    def dfs(self, i, j, part):
        self.grid[i][j] = part
        checkList = [[max(i-1, 0), j], [i, max(j-1, 0)], [min(i+1, self.rows), j], [i, min(j+1, self.cols)]]
        for ind in checkList:
            if type(self.grid[ind[0]][ind[1]]) == str and self.grid[ind[0]][ind[1]] == '1':
                self.dfs(ind[0], ind[1], part)
            else:
                continue

    def numIslands(self, grid):
        if grid == []:
            return 0
        
        self.grid = grid.copy()
        self.rows = len(self.grid) - 1
        self.cols = len(self.grid[0]) - 1
        
        part = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if type(self.grid[i][j]) == str and self.grid[i][j] == '1':
                    self.dfs(i, j, part)
                    part += 1
                else:
                    continue
        return part

if __name__ == "__main__":
    grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
    #grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
    s = Solution()
    res = s.numIslands(grid)