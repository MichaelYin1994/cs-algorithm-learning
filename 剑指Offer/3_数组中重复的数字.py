# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:23:16 2018

@author: XPS13
"""
# 不停的和numbers[ind]的位置的数互换位置，直到遍历完位置
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        if len(numbers) == 0:
            return False
        for ind, item in enumerate(numbers):
            if ind == item:
                continue
            else:
                while (ind != numbers[ind]):
                    if numbers[ind] == numbers[numbers[ind]]:
                        duplication[0] = numbers[ind]
                        return True
                    else:
                        tmp = numbers[numbers[ind]]
                        numbers[numbers[ind]] = numbers[ind]
                        numbers[ind] = tmp
        return False
