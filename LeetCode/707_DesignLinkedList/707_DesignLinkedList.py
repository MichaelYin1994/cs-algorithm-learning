# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:44:37 2019

@author: XPS13
"""
import fastdtw
class Node(object):
    def __init__(self, next_node, val):
        self.next = next_node
        self.val = val
'''
FAILED:
    Singly Linked List Version.
'''
class MyLinkedList(object):

    def __init__(self):
        # Initializing a new node and let the self.curr
        # points to this node.
        new_node = Node(None, None)
        
        # Parameters
        self.head, self.tail = new_node, new_node
        self.num_of_nodes = 0
        
        
    def get(self, index):
        if self.num_of_nodes == 0:
            return -1
        if (index < 0) or (index > (self.num_of_nodes - 1)):
            return -1
        
        # Start indexing the target node
        count_var, pointer = 0, self.head
        while(count_var != index):
            count_var += 1
            pointer = pointer.next
        return pointer.val
        
        
    def addAtHead(self, val):
        # Initizing a new node to store the value
        new_node = Node(None, None)
        new_node.next, new_node.val = None, val
        
        # Start adding the node
        if self.num_of_nodes == 0:
            self.head.val = new_node.val
        else:
            tmp_node = self.head
            self.head = new_node
            self.head.next = tmp_node
        
        # Adding an element
        self.num_of_nodes += 1
        
        
    def addAtTail(self, val):
        # Initizing a new node to store the value
        new_node = Node(None, None)
        new_node.next, new_node.val = None, val
        
        # Start adding the node
        if self.num_of_nodes == 0:
            self.tail.val = new_node.val
        else:
            self.tail.next = new_node
            self.tail = self.tail.next
        
        # Adding an element
        self.num_of_nodes += 1
        
        
    def addAtIndex(self, index, val):
        # Check the condition
        if index <= 0:
            self.addAtHead(val)
            return
        if index == self.num_of_nodes:
            self.addAtTail(val)
            return
        if index > self.num_of_nodes:
            return
        
        # Start inserting the node
        pos_to_add = index - 1
        count_var, pointer = 0, self.head
        while(count_var != pos_to_add):
            count_var += 1
            pointer = pointer.next
        
        new_node = Node(None, None)
        new_node.val = val
        
        tmp_node = pointer.next
        pointer.next = new_node
        new_node.next = tmp_node
        
        # Adding an element
        self.num_of_nodes += 1
        
        
    def deleteAtIndex(self, index):
        # Check condition
        if self.num_of_nodes == 0:
            return
        if index < 0 or index >= self.num_of_nodes:
            return        
        if index == 0:
            self.head = self.head.next
            self.num_of_nodes -= 1
            return

        # Start deleting the node
        pos_to_del = index - 1
        count_var, pointer = 0, self.head
        
        while(count_var != pos_to_del):
            count_var += 1
            pointer = pointer.next
        
        if pos_to_del == (self.num_of_nodes - 1):
            self.tail = pointer
            self.pointer.next = None
        else:
            pointer.next = pointer.next.next
        
        # Adding an element
        self.num_of_nodes -= 1