# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:49:25 2018

@author: XPS13
"""

# 回文序列查找：
# 遍历字符串，对于每一个字符串，分奇回文序列和偶回文序列进行讨论：若是奇回文序列，便
# 使用下标i-1到i+1作为输入，偶回文序列则是i到i+1。注意返回的是s[left+1:(right)]。
class Solution(object):
    def find(self, s, left, right):
        while(left >= 0 and right < len(s) and s[left] == s[right]):
            left -= 1
            right += 1
        print(s[left+1:(right)])
        return s[left+1:(right)]
    
    def longestPalindrome(self, s):
        if len(s) <= 1:
            return s
        ret = ""
        for ind, c in enumerate(s):
            # odd
            tmp_odd = self.find(s, ind-1, ind+1)
            if len(tmp_odd) > len(ret):
                ret = tmp_odd
            # even
            tmp_even = self.find(s, ind, ind+1)
            if len(tmp_even) > len(ret):
                ret = tmp_even
        return ret
string = "babad"
s = Solution()
print(s.longestPalindrome(string))