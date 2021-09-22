# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:54:12 2018

@author: XPS13
"""
# 题目意义在于把各个情况都要考虑全？至少暂时是这么理解的。比如base < 0的情况，base == 0
# 的情况等等，然后采用递归的方式做计算。

class Solution(object):
    def myPow(self, x, n):
        # base小于0，递归的计算1/x * 1/x * 1/x ...... * 1/x等。
        if n < 0:
            return 1/x * self.myPow(1/x, -(n+1))
        if n == 0:
            return 1
        if n == 2:
            return x**2
        
        # 判断奇数偶数，偶数可以直接递归，奇数需要base先转为偶数再递归
        if n % 2 == 0:
            return self.myPow(self.myPow(x, 2), n/2)
        if n % 2 == 1:
            return x * self.myPow(self.myPow(x, 2), (n-1)/2)