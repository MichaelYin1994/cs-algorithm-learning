# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:43:54 2018

@author: XPS13
"""
'''
Consider the following linked list, where E is the cylce entry and X, the crossing point of fast and slow.
        H: distance from head to cycle entry E
        D: distance from E to X
        L: cycle length
                          _____
                         /     \
        head_____H______E       \
                        \       /
                         X_____/   
        
        If fast and slow both start at head, when fast catches slow, slow has traveled H+D and fast 2(H+D). 
        Assume fast has traveled n loops in the cycle, we have:
        2H + 2D = H + D + nL  -->  H + D = nL  --> H = nL - D
        Thus if two pointers start from head and X, respectively, one first reaches E, the other also reaches E. 
        In my solution, since fast starts at head.next, we need to move slow one step forward in the beginning of part 2
'''
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return None
        pSlow = head
        pFast = head.next
        circleFlag = 0
        while(pFast != None and pFast.next != None):
            if pSlow == pFast:
                circleFlag = 1
                break
            else:
                pSlow = pSlow.next
                pFast = pFast.next.next
        if circleFlag == 0:
            return None
        pFast = pFast.next
        pSlow = head
        while(True):
            if pSlow == pFast:
                return pFast
            else:
                pSlow = pSlow.next
                pFast = pFast.next
    