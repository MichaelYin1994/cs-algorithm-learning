# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:21:38 2018

@author: XPS13
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 递归
class Solution(object):
    def __reverseList(self, curr, currNext):
        if currNext is None:
            return curr
        tmp = currNext.next
        currNext.next = curr
        return self.__reverseList(currNext, tmp)
        
    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        head = self.__reverseList(None, head)
        return head

# 非递归
class Solution_1(object):
    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        
        prev = None
        curr = head
        while(curr is not None):
            node = curr.next
            curr.next = prev
            prev = curr
            curr = node
        return prev
    
if __name__ == "__main__":
    head = ListNode(1)
    p1 = ListNode(2)
    p2 = ListNode(3) 
    p3 = ListNode(4)
    head.next = p1 
    p1.next = p2
    p2.next = p3
    s = Solution()
    node = s.reverseList(head)