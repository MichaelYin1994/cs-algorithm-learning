# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:18:32 2018

@author: XPS13
"""
def calc_maxium_length(s):
    stringDict = {}
    maxLength = 0
    start = 0
    
    for ind, c in enumerate(s):
        if c in stringDict:
            maxLength = max(maxLength, ind-start)
            start = max(stringDict[c] + 1, start)
        stringDict[c] = ind
    return max(maxLength, len(s)-start)

class Solution_1(object):
    def lengthOfLongestSubstring(self, s):
        if len(s) == 0:
            return 0
        elif len(s) == 1:
            return 1

        front, maxLength = 0, 0
        hashTab = {}
        for rear, c in enumerate(s):
            if c not in hashTab:
                hashTab[c] = rear 
            else:
                front = hashTab[c] + 1
                # 始终维护hashTab是front到rear的这一段
                hashTab = {s[i]: i for i in range(front, rear)}
                hashTab[c] = rear
            if len(hashTab) > maxLength:
                maxLength = len(hashTab)
        return  maxLength

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        if len(s) == 0:
            return 0
        elif len(s) == 1:
            return 1
        
        start, maxLength, hashTab = 0, 0, {}
        for ind, c in enumerate(s):
            if c not in hashTab:
                hashTab[c] = ind
            else:
                hashTab[c], start = ind, hashTab[c] + 1
            maxLength = max(ind - start + 1, maxLength)
        return maxLength

if __name__ == "__main__":
    #string = "bbutbld"
    string = "bbbbbbbbb"
    string = "au"
    string = "pwwkew"
    string = "abcabcab"
    string = "abba"
    s = Solution()
    ret = s.lengthOfLongestSubstring(string)
    