# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:00:15 2019

@author: XPS13
"""

class Solution(object):
    def coinChange(self, coins, amount):
        INF = float("inf")
        dp = [0] + [INF] * amount
        
        for i in range(1, amount + 1):
            dp[i] = min([dp[i - coin] + 1 if i - coin >= 0 else INF for coin in coins])
            
        return dp[amount] if dp[amount] != INF else -1
if __name__ == "__main__":
    coins = [1, 2, 5]
    amount = 11
    
    s = Solution()
    ret = s.coinChange(coins, amount)