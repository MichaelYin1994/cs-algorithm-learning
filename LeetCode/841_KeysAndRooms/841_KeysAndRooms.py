# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:11:55 2019

@author: XPS13
"""

class Solution(object):
    def depth_first_search(self, rooms, visitedPos, marked):
        # Marked the room
        marked[visitedPos] = True
        
        # Search the locked room
        need2visit = rooms[visitedPos]
        for r in need2visit:
            if marked[r] != True:
                self.depth_first_search(rooms, r, marked)
    
    def canVisitAllRooms(self, rooms):
        if len(rooms) == 0:
            return False
        
        # Initializing the marked array
        marked = [False] * len(rooms)
        marked[0] = True
        
        # Depth first search for each room
        need2visit = rooms[0]
        for r in need2visit:
            if marked[r] != True:
                self.depth_first_search(rooms, r, marked)
        
        if sum(marked) == len(rooms):
            return True
        else:
            return False