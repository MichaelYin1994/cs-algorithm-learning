# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:54:18 2019

@author: XPS13
"""
class Solution(object):
    def numRollsToTarget(self, d, f, target):
        if d == 1 and target <= f:
            return 1
        elif d == 1 and target > f:
            return 0
        else:
            ret = 0
            tmp = self.__numRollsToTarget(3, d, f, target)
            ret += tmp
        return ret
    
    def __numRollsToTarget(self, firstDice, d, f, target):
        currLevel, currLevelVals = 1, [firstDice]
        ret = 0
        while(currLevelVals and currLevel < d):
            # Possible value of the next level
            nextLevelVals = []
            for i in currLevelVals:
                nextLevelVals.extend([i + diceNum for diceNum in range(1, f+1)])
            
            # Pruning
            tmp = []
            for i in nextLevelVals:
                if i == target:
                    ret += 1
                elif i < target:
                    tmp.append(i)
            currLevel, currLevelVals = currLevel + 1, tmp
        return ret

if __name__ == "__main__":
    s = Solution()
    ret = s.numRollsToTarget(d=2, f=6, target=7)