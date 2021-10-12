# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:51:59 2018

@author: XPS13
"""
# 动态规划思想，stairs[i] = stairs[i-1] + stairs[i-2]
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1 or n == 2:
            return n
        
        memory = [0, 1, 2]
        for i in range(3, n+1):
            memory.append(memory[i-1] + memory[i-2])
        return memory[-1]