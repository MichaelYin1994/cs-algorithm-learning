# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:30:18 2019

@author: XPS13
"""

class Solution(object):
    def isValid(self, s):
        if len(s) % 2 != 0:
            return False
        if len(s) == 0:
            return True
        
        stack = []
        left2right = {"[":"]", "{":"}", "(":")"}
        right2left = {"]":"[", "}":"{", ")":"("}
        
        for c in s:
            if c in left2right.keys():
                stack.append(c)
            elif c in right2left.keys():
                bracketDict = {"[":0, "{":0, "(":0, "]":0, "}":0, ")":0}
                left = right2left[c]
                
                # 遍历栈，直到找到对应的左括号
                while(stack):
                    tmp = stack.pop()
                    if tmp == left:
                        break
                    else:
                        bracketDict[tmp] += 1
                        
                # 检查出栈的括号数量
                for left in left2right.keys():
                    if bracketDict[left] != bracketDict[left2right[left]]:
                        return False
                
        return False if stack else True # 检查栈是否为空