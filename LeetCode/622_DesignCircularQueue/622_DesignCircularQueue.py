# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:19:46 2018

@author: XPS13
"""

class Node(object):
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None
        
class MyCircularQueue(object):
    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self._head = None
        self._tail = None
        self._size = 0
        self._k = k
        
    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        # Add a new node at the front of the queue
        if value == None or self._size == self._k:
            return False
        
        newNode = Node(value)
        if self._head == None:
            self._head = newNode
            self._tail = newNode
            newNode.prev = newNode
            newNode.next = newNode
        else:
            newNode.next = self._head
            self._head.prev = newNode
            self._tail.next = newNode
            newNode.prev = self._tail
            
        self._head = newNode
        self._size += 1
        return True
    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self._size == 0:
            return False
        elif self._size == 1:
            self._head = None
            self._tail = None
        else:
            self._tail.prev.next = self._head
            self._head.prev = self._tail.prev
            self._tail = self._tail.prev
        self._size -= 1
        return True
        
        
    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self._tail == None:
            return -1
        else:
            return self._tail.val

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self._head == None:
            return -1
        else:
            return self._head.val

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        if self._size == 0:
            return True
        else:
            return False

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        if self._size == self._k:
            return True
        else:
            return False


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()