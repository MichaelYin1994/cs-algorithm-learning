# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:13:19 2019

@author: XPS13
"""
class Solution(object):
    def fizzBuzz(self, n):
        return ["Fizz" * (i % 3 == 0) + "Buzz" * (i % 5 == 0) + str(i) * (i % 3 != 0 and i % 5 != 0) for i in range(1, n+1)]
    
    # 更加优雅的解法
    def fizzBuzz_discussion(self, n):
        return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]