# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 03:00:31 2018

@author: XPS13
"""
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def deleteDuplicates(self, head):
        if head == None or head.next == None:
            return head
        # dummy节点与pre节点一开始指向同一个节点，并且dummy与pre存储的
        # 都是指针的值，指针指向同一个节点
        dummy = pre = ListNode(None)
        
        # 先让dummy.next为head节点的地址
        dummy.next = head
        while(head != None and head.next != None):
            # 判断条件
            if head.val == head.next.val:
                # 若是条件满足了，一直让head向后跳，pre节点保存的是前一个不同于head.val的
                # 节点的地址
                while(head != None and head.next != None and head.val == head.next.val):
                    head = head.next
                head = head.next
                # pre.next赋值头指针，为的是让dummy始终指向头指针
                pre.next = head
            else:
                # 这里pre = pre.next，目的是让pre一直指向head的前一个指针
                pre = pre.next
                head = head.next
        return dummy.next
