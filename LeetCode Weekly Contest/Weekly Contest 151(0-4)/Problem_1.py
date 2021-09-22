# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:09:20 2019

@author: XPS13
"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
#class Solution(object):
#    def removeZeroSumSublists(self, head):
#        if head is None:
#            return None
#        
#        # Save the head
#        dummyHead = ListNode(head.val)
#        dummyHead.next = head.next
#        nums = []
#        while(head != None):
#            nums.append(head.val)
#        head = head.next
#        
#        # Preform merging operation
#        ans, currSum = [], 0
#        for i in range(len(nums)):
#            curr = nums[i]
#            if curr + currSum == 0:
#                tmpSum = 0
#                while(tmpSum == -curr):
#                    tmpSum += ans.pop()
#                    
#        
#        return currSum
        

class Solution(object):
    def removeZeroSumSublists(self, nums):
        if len(nums) == 0:
            return None
        
        # Preform merging operation
        ans, currSum = [], 0
        for i in range(len(nums)):
            curr = nums[i]
            if len(ans) == 0:
                ans.append(curr)
                currSum += curr
            if curr + currSum == 0:
                tmpSum = 0
                while(tmpSum == -curr):
                    tmpSum += ans.pop()
                currSum -= tmpSum
            elif curr + ans[-1] == 0:
                currSum -= ans.pop()
            else:
                ans.append(curr)
                currSum += curr
        
        return currSum

if __name__ == "__main__":
    s = Solution()
    