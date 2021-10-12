# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 01:52:19 2018

@author: XPS13
"""

class Solution(object):
    def isPalindrome(self, x):
        if x < 0:
            return False
        elif x < 10:
            return True
        stack = []
        while(x != 0):
            low = x % 10
            stack.append(low)
            x = x // 10
        low = 0
        high = len(stack) - 1
        while(low <= high):
            if stack[low] == stack[high]:
                low += 1
                high -= 1
            else:
                return False
        return True

s = Solution()
res = s.isPalindrome(121)