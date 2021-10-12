# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 10:54:30 2018

@author: XPS13
"""
# 使用快慢指针。若是链表有环，快指针每次走两步，慢指针每次走一步，每次他们之间的间距都
# 会减1（或者说是加1，因为是环所以无所谓），那么迟早他们会相遇。
class Solution(object):
    def hasCycle(self, head):
        if head == None:
            return False
        elif head.next == None:
            return False
        
        pSlow = head
        pFast = head.next.next
        # 可能快指针已经为None了，所以while循环需要先判断一下。
        # 测试输入[1, 2]
        while(pFast != None):
            if pSlow == pFast:
                return True
            if pFast.next != None:
                pSlow = pSlow.next
                pFast = pFast.next.next
            else:
                return False
        return False