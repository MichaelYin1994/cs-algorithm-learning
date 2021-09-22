# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:32:16 2018

@author: XPS13
"""

# 一个更加优雅的解答，使用了Python的zip函数与*解引用操作符。
# a = ['flow', 'flight', 'flee']
# for ind, item in enumerate(zip(*a)):
#   print(item)
# ('f', 'f', 'f')
# ('l', 'l', 'l')
# ('o', 'i', 'e')
# ('w', 'g', 'e')
# 随后判读set之后的元素的个数。
class Solution:
    # @return a string
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
            
        for i, letter_group in enumerate(zip(*strs)):
            if len(set(letter_group)) > 1:
                return strs[0][:i]
        else:
            return min(strs)
        
# 自己的解法，暴力算法
class Solution_my(object):
    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ""
        compare = strs[0]
        common = ""
        flag = True
        for ind, c in enumerate(compare):
            for word in strs[1:]:
                if ind <= len(word) - 1 and word[ind] == c:
                    continue
                else:
                    flag = False
                    break
            if flag == False:
                break
            else:
                common += c
        return common