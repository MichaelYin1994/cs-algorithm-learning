# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:24:10 2019

@author: XPS13
"""
'''
# Solution 1: two pointer
class Solution:
    def intersect(self, nums1, nums2):
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        
        # Pre sort the array
        nums1.sort()
        nums2.sort()
        
        # Initializing the params
        i = j = 0
        length_1 = len(nums1)
        length_2 = len(nums2)
        res = []
        
        while(i != length_1 and j != length_2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                res.append(nums1[i])
                i += 1
                j += 1
        return res
'''
# Solution 2: Hash table
class Solution_hash:
    def intersect(self, nums1, nums2):
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        
        length_1 = len(nums1)
        length_2 = len(nums2)
        
        # Create a hash table for each of the array
        hashTab_1 = {}
        hashTab_2 = {}
        for i in nums1:
            if i not in hashTab_1:
                hashTab_1[i] = 1
            else:
                hashTab_1[i] += 1
        
        for i in nums2:
            if i not in hashTab_2:
                hashTab_2[i] = 1
            else:
                hashTab_2[i] += 1
        
        res = []
        if length_1 < length_2:
            for key in hashTab_1:
                if key in hashTab_2:
                    res.extend([key] * min(hashTab_1[key], hashTab_2[key]))
        else:
            for key in hashTab_2:
                if key in hashTab_1:
                    res.extend([key] * min(hashTab_1[key], hashTab_2[key]))
        return res
    
# Another more smart hash solution.
class Solution:
    def intersect(self, nums1, nums2):
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        
        # Create a hash table for each of the array
        res = []
        hashTab = {}
        for i in nums1:
            if i not in hashTab:
                hashTab[i] = 1
            else:
                hashTab[i] += 1
        
        for i in nums2:
            if i in hashTab and hashTab[i] > 0:
                res.append(i)
                hashTab[i] -= 1
        return res

if __name__ == "__main__":
#    nums1 = [1, 2, 2, 1]
#    nums2 = [2, 2]
    
    nums1 = [4,9,5]
    nums2 = [9,4,9,8,4]
    s = Solution_hash()
    res = s.intersect(nums1, nums2)