# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:35:36 2019

@author: XPS13
"""

class Solution(object):
    def groupAnagrams(self, strs):
        if len(strs) == 0:
            return []
        elif len(strs) == 1:
            return [strs]
        
        # Digitizing the word
        hashTab = {}
        for ind, word in enumerate(strs):
            tmp = [ 0 ] * 26
            for c in word:
                tmp[ord(c) - 97] += 1
            
            # Transfer the bits to Character 
            strTmp = ""
            for bit in tmp:
                strTmp += str(bit)
            
            # Check the hash table
            if strTmp not in hashTab:
                hashTab[strTmp] = [word]
            else:
                hashTab[strTmp].append(word)
        
        return hashTab.values

# Faster
class Solution_1(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        hashTab = {}
        for s in strs:
            sort_s = "".join(sorted(s))
            if sort_s not in hashTab:
                hashTab[sort_s] = [s]
            else:
                hashTab[sort_s].append(s)
    
        return hashTab.values()

if __name__ == "__main__":
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    s = Solution()
    ret = s.groupAnagrams(strs)