# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:19:56 2019

@author: XPS13
"""
# O(M + N) space solution
class Solution(object):
    def set_row_zeros(self, matrix, row):
        for i in range(self.cols):
            matrix[row][i] = 0
    
    def set_col_zeros(self, matrix, col):
        for i in range(self.rows):
            matrix[i][col] = 0
    
    def setZeroes(self, matrix):
        if len(matrix) == 0:
            return
        
        # Helper parameters
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        colZeros = set()
        rowZeros = set()
        
        # Iterative each element of the matrix
        for i in range(self.rows):
            for j in range(self.cols):
                if matrix[i][j] == 0:
                    rowZeros.add(i)
                    colZeros.add(j)
        
        # Set the elements to zero
        for row in rowZeros:
            self.set_row_zeros(matrix, row)
        for col in colZeros:
            self.set_col_zeros(matrix, col)

# O(1) solution: using the first row and col as indictor
class Solution_1(object):
    def setZeroes(self, matrix):
        if len(matrix) == 0:
            return
        
        colFlag, rowFlag, rows, cols = 1, 1, len(matrix), len(matrix[0])
        
        # Recording whether the first row and the first col contain 0
        for i in range(cols):
            if matrix[0][i] == 0:
                colFlag = 0
        
        for i in range(rows):
            if matrix[i][0] == 0:
                rowFlag = 0
        
        # Start from the second row and the second col
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        
        # Set zero from the bottom
        for i in range(rows-1, 0, -1):
            for j in range(cols-1, 0, -1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # Check the row and the col
        if colFlag == 0:
            for i in range(cols):
                matrix[0][i] = 0
        if rowFlag == 0:
            for i in range(rows):
                matrix[i][0] = 0

if __name__ == "__main__":
    matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    s = Solution()
    s.setZeroes(matrix)