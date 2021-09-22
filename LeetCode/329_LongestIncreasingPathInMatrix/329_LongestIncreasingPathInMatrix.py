# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:17:27 2019

@author: XPS13
"""

class Solution(object):
    def depth_first_search(self, i, j, path):
        # Early stop
        if self.longestPathMatrix[i][j] is not None:
            return self.longestPathMatrix[i][j] + path
        path = path + 1
        
        # CheckInd: left, right, up, down
        checkInd = [[max(0, i-1), j], [min(self.rows-1, i+1), j], [i, max(0, j-1)], [i, min(self.cols-1, j+1)]]
        longestPath = [1] * 4
        
        # dfs
        ind = 0
        for x, y in checkInd:
            if self.matrix[x][y] > self.matrix[i][j]:
                longestPath[ind] = self.depth_first_search(x, y, path)
            ind += 1
        self.longestPathMatrix[i][j] = max(longestPath)
        return self.longestPathMatrix[i][j] + path
        
    def find_longest_path(self, i, j):
        return self.depth_first_search(i, j, 0)
        
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0
        self.matrix = matrix
        self.longestPathMatrix = []
        self.rows, self.cols = len(matrix), len(matrix[0])
        for i in range(self.rows):
            self.longestPathMatrix.append([None] * self.cols)
        longestPath = 0
        
        # Find the longest path for each element
        for i in range(self.rows):
            for j in range(self.cols):
                self.longestPathMatrix[i][j] = self.find_longest_path(i, j)
                if self.longestPathMatrix[i][j] > longestPath:
                    longestPath = self.longestPathMatrix[i][j]
        return longestPath

if __name__ == "__main__":
    nums = [[9,9,4], [6,6,8], [2,1,1]]
    nums = [[3,4,5], [3,2,6], [2,2,1]] 
    s = Solution()
    ret = s.longestIncreasingPath(nums)
    
    

