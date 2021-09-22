# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:52:51 2020

@author: XPS13
"""
from collections import deque
class MonotonicQueue():
    """
    @Description:
    ---------
    Monotonic queue implementation using python deque class.

    @References:
    ---------
    [1] (Leetcode 239)https://leetcode.com/problems/sliding-window-maximum/description/
    [2] (Leetcode 239 solution tutorial)https://www.bilibili.com/s/video/BV1WW411C763
    [3] (Python deque)https://docs.python.org/zh-cn/3/library/collections.html#deque-objects
    """
    def __init__(self):
        self.queue = deque()

    def pop(self):
        return self.queue.popleft()

    def push(self, elem):
        while(len(self.queue) != 0 and elem > self.queue[-1]):
            self.queue.pop()
        self.queue.append(elem)

    def maximum(self):
        return self.queue[0]


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if len(nums) == k:
            return [max(nums)]
        ans, m_queue = [], MonotonicQueue()

        for i in range(len(nums)):
            m_queue.push(nums[i])
            if i - k + 1 >= 0:
                ans.append(m_queue.maximum())
                if nums[i - k + 1] == m_queue.maximum():
                    m_queue.pop()
        return ans


if  __name__ == "__main__":
    k = 6
    nums = [1, 3, -1, -3, 5, 3, 6, 7]

    k = 5
    nums = [4, 3, 2, 4, 0, 0, 0, 0]

    k = 2
    nums = [7, 2, 4]

    # [3,3,5,5,6,7]
    s = Solution()
    res = s.maxSlidingWindow(nums, k)