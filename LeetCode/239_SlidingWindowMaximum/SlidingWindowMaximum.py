# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:07:19 2021

@author: zhuoyin
"""

from collections import deque
class MonotonicQueue():
    def __init__(self):
        self.deque = deque()

    def push(self, x):
        self.deque.append(x)
        while(len(self.deque) and x > self.deque[0]):
            self.deque.popleft()

    def pop(self):
        pass

    def get_maximum(self):
        return self.deque[0]


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        if len(nums) == 1 or k == 1:
            return nums
        m_queue, window_maximum = MonotonicQueue(), []

        for ind, item in enumerate(nums):
            pass

            if ind + 1 >= k:
                window_maximum.append()


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