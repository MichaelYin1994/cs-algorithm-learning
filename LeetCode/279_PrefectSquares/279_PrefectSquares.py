# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:55:49 2019

@author: XPS13
"""

# Classic Dynamic Programming Problem
class Solution_1(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        if n == 0:
            return None
        elif n == 1:
            return 1
        
        # Create the dp matrix
        dp = [i for i in range(0, n + 1)]
        
        # Opitmal condition:
        # dp[i](Prefect Squares) = dp[i - j*j] + 1 while j*j < i
        # Maximum Prefect Squares == i itself( 1 + 1 + ... + 1 == i)
        for i in range(1, n + 1):
            j = 1
            while(j * j <= i):
                dp[i] = min(dp[i], dp[i - j*j] + 1)
                j += 1
        return dp[-1]
    
# Breadth-First-Search method for the solution
import collections
class Solution_2(object):
    def numSquares(self, n):
        # Unfriendly input
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        
        # Generating the square numbers, and early stop
        squareNumbers, i = [], 1
        while(i * i <= n):
            if i * i == n:
                return 1
            else:
                squareNumbers.append(i * i)
                i += 1
                
        # Breath first search for the target                
        queue, level, visited = collections.deque(), -1, set()
        queue.append(n)
        while(len(queue) != 0):
            currSize, level = len(queue), level + 1
            for i in range(currSize):
                node = queue.popleft()
                if node == 0:
                    return level
                for j in squareNumbers:
                    tmp = node - j
                    if tmp >= 0 and tmp not in visited:
                        queue.append(tmp)
                        visited.add(tmp)
        return level
        
if __name__ == "__main__":
    n = 7168
    s = Solution_2()
    ret = s.numSquares(n)
        
        
        
        
        
