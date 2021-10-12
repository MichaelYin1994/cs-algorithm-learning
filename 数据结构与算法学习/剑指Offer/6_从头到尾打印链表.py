# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:25:20 2018

@author: XPS13
"""
# 使用两个栈，实现队列。
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        stack_1 = []
        stack_2 = []
        while (listNode != None):
            stack_1.append(listNode.val)
            listNode = listNode.next
        while(len(stack_1) != 0):
            stack_2.append(stack_1.pop())
        return stack_2