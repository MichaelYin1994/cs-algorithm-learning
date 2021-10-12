# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 01:05:14 2018

@author: XPS13
"""

class Solution(object):
    def countAndSay(self, n):
        if n == 0 or n > 30:
            return None
        
        memory = ['1']
        count = 0
        while(count <= n-2):
            number = memory[count]
            left = number[0]
            
            ind = 0
            countTmp = 0
            res = ''
            
            for ind, right in enumerate(number):
                if left != right:
                    res += str(countTmp) + left
                    left = right
                    countTmp = 1
                elif left == right:
                    countTmp += 1
                    
            res  += str(countTmp) + left
            memory.append(res)
            count += 1
            
        return memory[-1]

if __name__ == "__main__":
    s = Solution()
    res = s.countAndSay(29)