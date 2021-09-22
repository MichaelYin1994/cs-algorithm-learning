# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:20:41 2019

@author: XPS13
"""

class Solution(object):
    def dailyTemperatures(self, T):
        ans = [0] * len(T)
        stack = []
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                curr = stack.pop()
                ans[curr] = i - curr
            stack.append(i)
        return ans

if __name__ == "__main__":
    T = [73, 74, 75, 71, 69, 72, 76, 73]
    s = Solution()
    ret = s.dailyTemperatures(T)