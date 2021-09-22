# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:09:48 2019

@author: XPS13
"""

class Solution(object):
    def within_range(self, number):
        if number < -(2**31) or number > 2**31 - 1:
            return False
        else:
            return True
        
    def reverse(self, x):
        if x == 0:
            return 0
        ret = 0
        flag = 0 if x < 0 else 1
        x = abs(x)
        if self.within_range(x):
            while(x != 0):
                ret = ret * 10 + x % 10 
                x = x // 10
        else:
            return 0
        # 注意检查输出的范围在range里面
        ret = ret if flag == 1 else -ret
        return ret if self.within_range(ret) else 0
        