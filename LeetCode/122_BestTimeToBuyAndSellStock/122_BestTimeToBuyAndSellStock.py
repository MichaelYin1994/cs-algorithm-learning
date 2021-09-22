# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:25:32 2019

@author: XPS13
"""

class Solution:
    def maxProfit(self, prices):
        if len(prices) == 0:
            return 0
        
        maxProfit = 0
        for i in range(1, len(prices)):
            if prices[i] >= prices[i-1]:
                maxProfit += prices[i] - prices[i-1]
        return maxProfit