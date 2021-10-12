# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:03:43 2019

@author: XPS13
"""
# 注意str的isnalsum方法，检测字符是否为数字或者字母
class Solution(object):
    def isPalindrome(self, s):
        if len(s) == 0:
            return True
        elif len(s) == 1:
            return True
        
        res = []
        for c in s:
            if c.isalnum() : res.append(c.lower())
                
        left, right = 0, len(res) - 1
        while(left <= right):
            if res[left] != res[right]:
                return False
            else:
                left += 1
                right -= 1
        return True