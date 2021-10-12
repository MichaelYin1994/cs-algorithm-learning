# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:21:25 2018

@author: XPS13
"""
# 左旋字符串，使用将字符串拼凑的方式进行旋转。
class Solution:
    def LeftRotateString(self, s, n):
        stringSize = len(s)
        if len(s) == 0:
            return ""
        elif len(s) == 1:
            return s
        # 分情况讨论问题，看看余数的情况
        if stringSize > n:
            return (s + s)[n:n+stringSize]
        elif stringSize == n:
            return s
        else:
            mod = n % stringSize
            return (s + s)[mod:mod+stringSize]