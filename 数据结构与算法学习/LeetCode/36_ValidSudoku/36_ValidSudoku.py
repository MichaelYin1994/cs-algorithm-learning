# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:57:03 2019

@author: XPS13
"""
# Navie Algorithm
class Solution:
    def row_check(self, board):
        for rowList in board:
            hashTab = {}
            for item in rowList:
                if item in hashTab:
                    return False
                elif item is not ".":
                    hashTab[item] = 1
        return True
    
    def column_check(self, board):
        for col in range(len(board[0])):
            hashTab = {}
            for row in range(len(board)):
                item = board[row][col]
                if item in hashTab:
                    return False
                elif item is not ".":
                    hashTab[item] = 1
        return True
    
    def square_check(self, board):
        for row in range(0, len(board), 3):
            for col in range(0, len(board), 3):
                hashTab = {}
                for i in range(0, 3):
                    for j in range(0, 3):
                        item = board[row+i][col+j]
                        if item in hashTab:
                            return False
                        elif item is not ".":
                            hashTab[item] = 1
        return True
        
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        if len(board) == 0:
            return False
        if self.row_check(board) is False:
            return False
        elif self.column_check(board) is False:
            return False
        else:
            return self.square_check(board)