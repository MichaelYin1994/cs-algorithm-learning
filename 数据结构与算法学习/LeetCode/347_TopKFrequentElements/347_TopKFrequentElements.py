# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:37:27 2019

@author: XPS13
"""

class Solution(object):
    def topKFrequent(self, nums, k):
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return nums
        
        # Counting the numbers
        count = {}
        for i in nums:
            if i not in count.keys():
                count[i] = 1
            else:
                count[i] += 1
        
        # Reverse hash
        freq = {}
        for key, val in zip(count.keys(), count.values()):
            if val not in freq:
                freq[val] = [key]
            else:
                freq[val].append(key)
        
        # Traversal from the rear
        ret = []
        for i in range(max(freq.keys()), 0-1, -1):
            if i in freq:
                ret.extend(freq[i])
        return [ret[i] for i in range(k)]

if __name__ == "__main__":
    nums = [1, 1, 1, 2, 2, 4]
    k = 3
    s = Solution()
    ret = s.topKFrequent(nums=nums, k=k)
    