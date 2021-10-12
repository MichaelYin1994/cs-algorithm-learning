# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:07:52 2019

@author: XPS13
"""

###############################################################################
###############################################################################
#class Solution(object):
#    '''
#    Solution 1:
#        《算法》第四版的深度优先搜索模板。
#        少check了一个条件！若是image[sr][sc] == newColor则可直接返回。
#    '''
#    def floodFill(self, image, sr, sc, newColor):
#        if len(image) == 0:
#            return image
#        elif image[sr][sc] == newColor:
#            return image
#        
#        # Initializing some basic params
#        self.rows, self.cols = len(image), len(image[0])
#        self.marked, self.newColor, self.color = [], newColor, image[sr][sc]
#        for i in range(self.rows):
#            self.marked.append([False] * self.cols)
#        
#        # Depth-first-search for the connected component
#        searchDirection = [[min(self.rows-1, sr+1), sc], [max(0, sr-1), sc], 
#                           [sr, min(self.cols-1, sc+1)], [sr, max(0, sc-1)]]
#        for i, j in searchDirection:
#            if image[i][j] == self.color:
#                self.depth_first_search([i, j], image)
#        
#        # Modified the initial position
#        image[sr][sc] = newColor
#        
#        return image
#    
#    def depth_first_search(self, coord=None, image=None):
#        sr, sc = coord[0], coord[1]
#        image[sr][sc], self.marked[sr][sc] = self.newColor, True
#        
#        searchDirection = [[min(self.rows-1, sr+1), sc], [max(0, sr-1), sc], 
#                           [sr, min(self.cols-1, sc+1)], [sr, max(0, sc-1)]]
#        for i, j in searchDirection:
#            if self.marked[i][j] is False and image[i][j] == self.color:
#                self.depth_first_search([i, j], image)

###############################################################################
###############################################################################
class Solution():
    '''
    Solution:
        不带有marked数组，BFS新范式，color参数充当了marked数组的角色。
    '''
    def floodFill(self, image, sr, sc, newColor):
        if len(image) == 0:
            return image
        elif image[sr][sc] == newColor:
            return image
        
        self.fill(image, sr, sc, newColor, image[sr][sc])
        return image
    
    def fill(self, image, sr, sc, newColor, color):
        if (sr < 0) or (sr > len(image) - 1) or (sc < 0) or (sc > len(image[0]) - 1) or (image[sr][sc] != color):
            return

        image[sr][sc] = newColor
        self.fill(image, sr-1, sc, newColor, color)
        self.fill(image, sc+1, sc, newColor, color)
        self.fill(image, sc, sr-1, newColor, color)
        self.fill(image, sc, sr+1, newColor, color)
        return


