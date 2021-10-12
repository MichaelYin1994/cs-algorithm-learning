# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:32:07 2018

@author: XPS13
"""
# 用一个辅助栈，检测辅助栈的栈顶元素，与出栈序列之间的差别。
class Solution:
    def IsPopOrder(self, pushV, popV):
        if pushV == None or popV == None:
            return False
        extraStack = []
        extraStack.append(pushV[0])
        del pushV[0]
        popInd = 0
        ret = True
        while(len(pushV) or len(extraStack)):
            if extraStack[-1] != popV[popInd] and pushV:
                extraStack.append(pushV[0])
                del pushV[0]
            elif extraStack[-1] == popV[popInd]:
                extraStack.pop()
                popInd += 1
            elif len(pushV) == 0:
                ret = False
                break
        return ret