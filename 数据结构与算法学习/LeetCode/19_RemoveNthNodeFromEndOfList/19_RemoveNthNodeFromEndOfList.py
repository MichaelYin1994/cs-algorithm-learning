# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:15:43 2018

@author: XPS13
"""

#Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

###############################################################################
# 解法1：移动值
# 思路不是移动节点，而是递归的移动结点的值（有点"Cheating"的嫌疑==）
class Solution_1:
    def removeNthFromEnd(self, head, n):
        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            # 当遇到被删除结点的前一个结点的时候
            # 将被删除结点的值赋值为它的前一个结点
            if i > n:
                node.next.val = node.val
            return i
        index(head)
        return head.next

###############################################################################
# 解法2：快慢指针
# 思路不是移动节点，而是递归的移动结点的值（有点"Cheating"的嫌疑==）
class Solution_2:
    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head

###############################################################################
# Solution 3: List 模拟队列
class Solution_3(object):
    def removeNthFromEnd(self, head, n):
        if head is None or head.next is None:
            return None
        
        # Save the head node
        dummyNode = ListNode(None)
        dummyNode.next = head
        
        fixSizeQueue = []
        linkedListSize = 0
        while(head != None):
            # Enqueue an element
            fixSizeQueue.append(head)
            linkedListSize += 1
            
            # Dequeue an element
            if len(fixSizeQueue) == n + 1:
                node = fixSizeQueue[0]
                fixSizeQueue = fixSizeQueue[1:]
            
            # Move the header one step
            head = head.next
        
        if n == linkedListSize:
            return dummyNode.next.next
        elif n == 1:
            node.next = None
            return dummyNode.next
        else:
            node.next = node.next.next
            return dummyNode.next

###############################################################################
class Solution(object):
    def removeNthFromEnd(self, head, n):
        if head == None:
            return []
        curr = head
        pioneer = head
        prev = ListNode(None)
        prev.next = curr
        while(pioneer != None and n != 0):
            pioneer = pioneer.next
            n -= 1
            
        while(pioneer != None):
            pioneer = pioneer.next
            curr = curr.next
            prev = prev.next
        
        # 分情况讨论，有可能移除的是头结点，返回下一个结点就好了
        # 其他的情况就是正常操作
        if curr == head:
            return curr.next
        else:
            prev.next = curr.next
            return head