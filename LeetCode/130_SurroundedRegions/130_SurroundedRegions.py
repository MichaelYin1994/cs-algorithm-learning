# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:42:03 2019

@author: XPS13
"""
class Solution(object):
    def depth_first_search(self, count, i, j, board):
        board[i][j] = count
        checkList = [[max(i-1, 0), j], [i, max(j-1, 0)], [min(i+1, self.rows), j], [i, min(j+1, self.cols)]]
        for m, n in checkList:
            if board[m][n] == "O":
                self.depth_first_search(count, m, n, board)
        
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if len(board) == 0:
            return
        
        # Initializing the dfs array
        self.rows, self.cols = len(board), len(board[0])
        
        # Depth first search for the connected set
        count = 1
        for i in range(self.rows):
            for j in range(self.cols):
                if board[i][j] == "X":
                    board[i][j] = 0
                elif board[i][j] == "O":
                    self.depth_first_search(count, i, j, board)
                    count += 1
                    
        # Check border
        noFliped = set()
        for i in range(self.rows):
            if board[i][0] != 0:
                noFliped.add(board[i][0])
            if board[i][-1] != 0:
                noFliped.add(board[i][-1])
        
        for i in range(self.cols):
            if board[0][i] != 0:
                noFliped.add(board[0][i])
            if board[-1][i] != 0:
                noFliped.add(board[-1][i])
        
        for i in range(self.rows):
            for j in range(self.cols):
                if board[i][j] not in noFliped:
                    board[i][j] = "X"
                else:
                    board[i][j] = "O"

if __name__ == "__main__":
    board = [["O","O"],["O","O"]]
    s = Solution()
    s.solve(board)