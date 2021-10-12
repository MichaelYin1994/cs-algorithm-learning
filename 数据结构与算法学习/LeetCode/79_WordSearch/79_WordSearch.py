# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:49:00 2019

@author: XPS13
"""

class Solution(object):
    def helper(self, word, i, j):
        if word == "":
            return True
        c, ret, self.marked[i][j] = word[0], False, True
        checkInd = [[max(i - 1, 0), j], [i, max(j - 1, 0)], [min(i + 1, self.rows - 1), j], [i, min(j+1, self.cols - 1)]]
        for m, n in checkInd:
            if self.board[m][n] == c and self.marked[m][n] is False:
                ret = self.helper(word[1:], m, n)
            if ret:
                return True
        self.marked[i][j] = False
        return ret
    
    def exist(self, board, word):
        if board == []:
            return False
        
        self.rows, self.cols = len(board), len(board[0])
        self.posHash, self.board, ret = {}, board, False
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] not in self.posHash:
                    self.posHash[board[i][j]] = []
                    self.posHash[board[i][j]].append([i, j])
                else:
                    self.posHash[board[i][j]].append([i, j])
        
        if word[0] in self.posHash:
            for i, j in self.posHash[word[0]]:
                self.marked = [[False] * self.cols for i in range(self.rows)]
                ret = self.helper(word[1:], i, j)
                if ret:
                    return True
        return ret

if __name__ == "__main__":
    board = [["a","a"]]
    word = "aa"
    s = Solution()
    ret = s.exist(board, word)
    