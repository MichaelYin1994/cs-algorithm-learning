# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:27:45 2018

@author: XPS13
"""

'''
两趟while测试链表长度，不等长的话对其再判断。
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
'''
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        if headA == None or headB == None:
            return None
        currA, currB = headA, headB
        
        nodeNumsA = 0
        nodeNumsB = 0
        while(currA != None):
            currA = currA.next
            nodeNumsA += 1
        while(currB != None):
            currB = currB.next
            nodeNumsB += 1
        
        ret = None
        if nodeNumsA > nodeNumsB:
            while(nodeNumsB != nodeNumsA):
                headA = headA.next
                nodeNumsA -= 1
        elif nodeNumsA < nodeNumsB:
            while(nodeNumsB != nodeNumsA):
                headB = headB.next
                nodeNumsB -= 1
        
        while( headA != None and headB != None):
            if headA != headB:
                headA = headA.next
                headB = headB.next
            else:
                return headA
        return ret