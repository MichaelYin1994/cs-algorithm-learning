# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:56:26 2018

@author: XPS13
"""

#Input: 0->1->2->NULL, k = 4
#Output: 2->0->1->NULL
#Explanation:
#rotate 1 steps to the right: 2->0->1->NULL
#rotate 2 steps to the right: 1->2->0->NULL
#rotate 3 steps to the right: 0->1->2->NULL
#rotate 4 steps to the right: 2->0->1->NULL

class Solution(object):
    def rotateRight(self, head, k):
        if head == None:
            return None
        elif head.next == None:
            return head
        
        headSave = head
        # 头结点为第一个节点
        countSave = 1
        
        # 遍历链表获取链表长度，并构建循环链表
        while(head.next != None):
            head = head.next
            countSave += 1
        head.next = headSave
        
        # 计算应该算到第几个节点，+ 1代表算到前一个节点
        k = k % countSave + 1
        count = countSave
        head = headSave
        while(count != k):
            head = head.next
            count -= 1
        
        ret = head.next
        head.next = None
        return ret