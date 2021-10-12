# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:07:55 2019

@author: XPS13
"""

class Solution(object):
    def evalRPN(self, tokens):
        if len(tokens) == 1:
            return int(tokens[0])
        
        stack, ops = [], ["*", "+", "/", "-"]
        for c in tokens:
            if c not in ops:
                stack.append(int(c))
            else:
                rear = stack.pop()
                front = stack.pop()
                if c is "*":
                    stack.append(front * rear)
                elif c is "+":
                    stack.append(front + rear)
                elif c is "-":
                    stack.append(front - rear)
                else:
                    stack.append(int(front / rear))
        return int(stack[0])

if __name__ == "__main__":
    tokens = ["18"]
    s = Solution()
    ret = s.evalRPN(tokens)