# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:09:01 2019

@author: XPS13
"""

'''
# 自己的不够优雅的代码：
# Hardcore判别，常规操作，注意进位
class Solution:
    def plusOne(self, digits):
        if len(digits) == 0:
            return [1]
        
        digits[-1] = digits[-1] + 1
        if digits[-1] < 10:
            return digits
        
        pos = len(digits) - 1
        carryFlag = 0
        while(pos >= 0):
            digits[pos] += carryFlag
            if digits[pos] >= 10:
                carryFlag = 1
                digits[pos] = 0
            else:
                # 例如test输入[8, 9, 9, 9]，这里为了防止
                carryFlag = 0
            pos -= 1
        
        if carryFlag == 1:
            digits.insert(0, 1)
            return digits
        else:
            return digits
'''

class Solution:
    def plusOne(self, digits):
        if len(digits) == 0:
            return [1]
        
        pos = len(digits) - 1
        carryFlag = 1
        while(pos >= 0):
            digits[pos] += carryFlag
            # Early stop，若是有小于9的早就返回了
            if digits[pos] <= 9:
                return digits
            digits[pos] = 0
            pos -= 1
        
        digits.insert(0, 1)
        return digits
