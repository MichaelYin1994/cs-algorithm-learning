# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:10:04 2018

@author: XPS13
"""
# 测试输入[1, 1, 1, 1]注意
class Solution(object):
    def deleteDuplicates(self, head):
        if head == None or head.next == None:
            return head
        curr = head.next
        prev = head
        while(curr != None):
            if curr.val == prev.val:
                prev.next = curr.next
                curr = curr.next
                continue
            prev = curr
            curr = curr.next
        return head