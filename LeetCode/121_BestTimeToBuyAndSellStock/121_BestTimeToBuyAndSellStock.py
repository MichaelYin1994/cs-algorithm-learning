# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:22:34 2019

@author: XPS13
"""
class Solution:
    def maxProfit(self, prices):
        if len(prices) == 0:
            return 0
        
        # 过去的最低价格
        minPrice = prices[0]
        # 到目前为止的最高收益
        maxProfit = 0
        for i in prices:
            if i < minPrice:
                minPrice = i
            profitTmp = i - minPrice
            if profitTmp > maxProfit:
                maxProfit = profitTmp
        return maxProfit