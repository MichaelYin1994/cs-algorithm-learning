# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 02:34:58 2018

@author: XPS13
"""
# 没有人规定List里只能存储值，也可以存储元组，元组包含当前的值与当前的最小值
class MinStack(object):
    def __init__(self):
        self._array = []
        self._currMin = None
        
    def push(self, x):
        if x < self._currMin or self._currMin == None:
            self._array.append((x, x))
            self._currMin = x
        else:
            self._array.append((x, self._currMin))

    def pop(self):
        # 注意边界条件，不要让数组越界
        if len(self._array) <= 1:
            self._currMin = None
            self._array = []
        else:
            self._currMin = self._array[-2][1]
            del self._array[-1]
        
    def top(self):
        # 同样是注意边界条件
        if len(self._array) == 0:
            return None
        else:
            return self._array[-1][0]
    
    def getMin(self):
        return self._currMin
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()