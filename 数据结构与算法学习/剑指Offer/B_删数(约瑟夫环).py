# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:21:28 2019

@author: XPS13
"""

import sys
class Solution():
    def deleteNumber(self, number=None, gap=2):
        if number == 1:
            return 0
        if number == 2:
            return 1
         
        # Initializing the circular linked list
        linkedList = [[i-1, i, i+1] for i in range(number)]
        linkedList[0][0], linkedList[-1][2] = number - 1, 0
        deleteCount = 0
         
        # Start deleting the element
        curr = 0
        while(True):
            gapCount = gap
            while(gapCount != 0):
                curr = linkedList[curr][2]
                gapCount -= 1
            if deleteCount == (number - 1):
                return linkedList[curr][1]
            else:
                prevNode = linkedList[linkedList[curr][0]]
                nextNode = linkedList[linkedList[curr][2]]
                 
                prevNode[2] = nextNode[1]
                nextNode[0] = prevNode[1]
                deleteCount += 1
                curr = nextNode[1]
 
if __name__ == "__main__":
#    number = int(sys.stdin.readline())
    number = 216
    s = Solution()
    res = s.deleteNumber(number=number, gap=2)
    print(res)