# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:06:58 2018

@author: Administrator
"""

class Solution:
    def binarySearch(self, target):
        low = 0
        high = target//2+1-1
        
        while(low <= high):
            mid = low + (high - low) // 2
            if (mid*mid > target):
                high = mid - 1
            if (mid*mid < target):
                low = mid + 1
            if (mid*mid == target):
                return mid
        return low - 1
    
    def mySqrt(self, X):
        if X == 0:
            return 0
        elif X == 1:
            return 1
        elif X == 2:
            return 1
        
        res = self.binarySearch(X)
        return res

if __name__ == '__main__':
    X = 2147395599
    s = Solution()
    res = s.mySqrt(X)