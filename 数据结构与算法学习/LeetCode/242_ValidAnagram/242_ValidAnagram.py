# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:22:57 2019

@author: XPS13
"""
# 对于各种类型字符的异构词都成立
class Solution(object):
    def isAnagram(self, s, t):
        if len(s) == 0 and len(t) == 0:
            return True
        elif len(s) != len(t):
            return False
        
        hashTab_1 = {}
        hashTab_2 = {}
        
        for c_1, c_2 in zip(s, t):
            if c_1 not in hashTab_1:
                hashTab_1[c_1] = 1
            else:
                hashTab_1[c_1] += 1
            
            if c_2 not in hashTab_2:
                hashTab_2[c_2] = 1
            else:
                hashTab_2[c_2] += 1
        
        if len(hashTab_1) == len(hashTab_2):
            for c in hashTab_1.keys():
                if c not in hashTab_2 or hashTab_2[c] != hashTab_1[c]:
                    return False
        else:
            return False
        return True
        