# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:12:10 2019

@author: XPS13
"""

class Solution:
    def rotate_matrix(self, n):
        # Step 1: up, right exchange
        for i in range(n, self.matSize-n):
            self.matrix[n][i], self.matrix[i][self.matSize-n] = self.matrix[i][self.matSize-n], self.matrix[n][i]
        
        # Step 2: up, down exchange
        for i in range(n, self.matSize-n):
            self.matrix[n][i], self.matrix[self.matSize-n][i] = self.matrix[self.matSize-n][i], self.matrix[n][i]
        
        # step 3: up, left exchange
        for i in range(n, self.matSize-n):
            self.matrix[n][i], self.matrix[i][n] = self.matrix[i][n], self.matrix[n][i]
            
    def rotate(self, matrix):
        if len(matrix) == 0:
            return []
        
        self.matrix = matrix
        self.matSize = len(matrix)
        for i in range(0, self.matSize//2):
            self.rotate_matrix(i)
if __name__ == "__main__":
    matrix = [[7,4,1],[8,5,2],[9,6,3]]
    s = Solution()
    s.rotate(matrix)