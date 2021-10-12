# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:40:32 2019

@author: XPS13
"""

'''
Solution 0: Error
'''
#class Solution(object):
#    def check(self, i, j, dp, matrix, currentLength):
#        # Check coordinate, check marked
#        if (i < 0) or (i > self.rows - 1) or (j < 0) or (j > self.cols - 1):
#            return float("inf")
#        elif dp[i][j] is True:
#            return float("inf")
#        
#        # Dynamic programming
#        if dp[i][j] != float("inf"):
#            return dp[i][j] + currentLength
#        elif matrix[i][j] == 0:
#            return currentLength
#        
#        # Possible shortest path
#        l_1 = self.check(i+1, j, dp, matrix, currentLength+1)
#        l_2 = self.check(i-1, j, dp, matrix, currentLength+1)
#        l_3 = self.check(i, j+1, dp, matrix, currentLength+1)
#        l_4 = self.check(i, j-1, dp, matrix, currentLength+1)
#        return min(l_1, l_2, l_3, l_4)
#        
#    
#    def updateMatrix(self, matrix):
#        if len(matrix) == 0:
#            return matrix
#        
#        # Initializing parameters
#        # For dp matrix: element 1: pathNums, element 2: visited?
#        self.rows, self.cols = len(matrix), len(matrix[0])
#        dp = []
#        for i in range(self.rows):
#            dp.append([float("inf")] * self.cols)
#        
#        # Check the 0s
#        for i in range(self.rows):
#            for j in range(self.cols):
#                dp[i][j] = self.check(i, j, dp, matrix, 0)
#        return dp

'''
Solution 1: TLE
'''
#from collections import deque
#class Solution():
#    def updateMatrix(self, matrix):
#        if len(matrix) == 0:
#            return matrix
#        
#        # Initializing the searching directions, queue and dp matrix
#        searchDir = [[1, 0], [-1, 0], [0, 1], [0, -1]]
#        numRows, numCols = len(matrix), len(matrix[0])
#        queue = deque()
#        
#        for i in range(len(matrix)):
#            for j in range(len(matrix[0])):
#                if matrix[i][j] == 0:
#                    queue.append([i, j])
#                else:
#                    matrix[i][j] = float("inf")
#        
#        # Start the queue inserting process
#        while(len(queue) != 0):
#            pos = queue.pop()
#            searchPos = [[pos[0] + tmp[0], pos[1] + tmp[1]] for tmp in searchDir]
#            dist = matrix[pos[0]][pos[1]]
#            for nextPos in searchPos:
#                # If the distance of the neighboor points is smaller than the 
#                # current dist + 1, then there is no need to continue.(Dijkstra)
#                if nextPos[0] < 0 or nextPos[0] >= numRows or nextPos[1] < 0 or nextPos[1] >= numCols or dist + 1 > matrix[nextPos[0]][nextPos[1]]:
#                    continue
#                else:
#                    matrix[nextPos[0]][nextPos[1]] = dist + 1
#                    queue.append(nextPos)
#        return matrix

'''
Solution 2
'''
class Solution():
    def updateMatrix(self, matrix):
        pass

if __name__ == "__main__":
    s = Solution()
    m = [[0,0,0], [0,1,0], [1,1,1]]
    ret = s.updateMatrix(m)