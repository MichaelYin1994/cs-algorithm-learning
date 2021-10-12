# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:14:47 2019

@author: XPS13
"""

class Solution(object):
    def in_order_traversal(self, currChar, currRes, remainDigits):
        currRes = currRes + currChar
        if len(remainDigits) == 0:
            self.ret.append(currRes)
        else:
            for c in self.digits2char[remainDigits[0]]:
                self.in_order_traversal(c, currRes, remainDigits[1:])
        
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        self.digits2char = {
            "1":"",
            "2":"abc",
            "3":"def",
            "4":"ghi",
            "5":"jkl",
            "6":"mno",
            "7":"pqrs",
            "8":"tuv",
            "9":"wxyz",
        }
        if len(digits) == 0:
            return []
        elif len(digits) == 1:
            return [c for c in self.digits2char[digits]]
        
        self.ret = []
        for c in self.digits2char[digits[0]]:
            self.in_order_traversal(c, "", digits[1:])
        
        return self.ret
        
if __name__ == "__main__":
    s = Solution()
    ret = s.letterCombinations("2237542")
    