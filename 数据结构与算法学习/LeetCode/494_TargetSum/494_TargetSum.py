# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:50:11 2019

@author: XPS13
"""

'''
Tips:
    1. TargetSum问题：
        Solution 1:(Dynamic Programming)
            典型的动态规划问题，可与背包问题联动。最先想到应该是基于递归的DFS，但是其
            时间复杂度应该是O(2^n)指数阶，n是nums包含的元素的个数；基于DP的思想可以对
            问题进行求解，时间复杂度O(n * sum(nums))。可以采用二维DP的方式，dp思路与最优性条件：
                -- dp[i][j]为利用nums[:i]相加（包括nums[i]），和等于j的方法的次数。
                -- 状态转移方程：dp[i][j] = dp[i-1][j - nums[i]] + dp[i-1][j + nums[i]]
                -- j的取值范围为[-sum(nums), sum(nums)]，为了索引方便应该是[0, 2 * sum(nums)]
                -- 初始条件：dp[0][0 + sum(nums)] == 0
                -- 注意边界条件
                -- return dp[-1][target + sum(nums)]
        
        Solution 2:
            相对上面的解答简化的dp。
            
        
    2. 0-1背包问题（一直没怎么弄明白）
        问题描述：
            pass

'''

class Solution(object):
    def findTargetSumWays(self, nums, S):
        # Prevent the invalid input and the out-of-range sum results
        if len(nums) == 0:
            return None
        if S > sum(nums)  or S < - sum(nums):
            return 0
        
        # Including 0 elements, sum to 0
        numSum = sum(nums)
        rows, cols = len(nums) + 1, numSum * 2 + 1
        
        # Create a dp matrix
        dp = []
        for i in range(rows):
            dp.append([0] * cols)
        dp[0][numSum] = 1
        
        # Optimal condition
        for i in range(1, rows):
            for j in range(cols):
                left, right = j - nums[i-1], j + nums[i-1]
                if left - numSum >= - numSum:
                    dp[i][j] += dp[i-1][left]
                if right - numSum <= numSum:
                    dp[i][j] += dp[i-1][right]
                    
        return dp[-1][numSum + S]

class Solution(object):
    def depth_first_search(self, nums=None, target=None, curr=None):
        if len(nums) == 0:
            if curr == target:
                self.total += 1
            return
        
        # In-depth search
        self.depth_first_search(nums[:-1], S, curr+nums[-1])
        self.depth_first_search(nums[:-1], S, curr-nums[-1])

    def findTargetSumWays(self, nums, S):
        if len(nums) == 0:
            return None
        
        # Global params
        self.total = 0
        
        # Depth first search
        self.depth_first_search(nums[:-1], S, nums[-1])
        self.depth_first_search(nums[:-1], S, -nums[-1])
        
        return self.total

if __name__ == "__main__":
    nums = [1, 1, 1, 1, 1]
    S = 3
    
    s = Solution()
    res = s.findTargetSumWays(nums, S)
    
