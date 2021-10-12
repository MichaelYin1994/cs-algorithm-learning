# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:37:49 2019

@author: XPS13
"""

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None or head.next.next is None:
            return head
        
        oddHead = ListNode(None)
        evenHead = ListNode(None)
        oddHead.next, evenHead.next = head, head.next, 
        odd, even, pioneer, count = head, head.next, head.next.next, 3
        while(pioneer is not None):
            if count % 2 != 0:
                odd.next = pioneer
                odd = odd.next
            else:
                even.next = pioneer
                even = even.next
            pioneer = pioneer.next
            count += 1
        even.next = None
        odd.next = evenHead.next
        return oddHead.next