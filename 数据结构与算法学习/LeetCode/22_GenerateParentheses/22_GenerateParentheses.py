# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:51:06 2018

@author: XPS13
"""
# 深度优先搜索 + 剪枝，if 语句充当剪枝的角色。
# 第一，注意剪枝的条件；第二，注意套深度优先搜索的模板
class Solution(object):
    def dfs(self, m, left, right):
        if left < right:
            return
        if (left == self.n) and (right == self.n):
            self.ret.append(m)
            return
        
        if left < self.n:
            self.dfs(m + "(", left + 1, right)
        if right < self.n:
            self.dfs(m + ")", left, right + 1)
        return
    
    def generateParenthesis(self, n):
        if n == 0:
            return []
        elif n == 1:
            return ["()"]
        elif n == 2:
            return ["()()", "(())"]
        self.n = n
        self.ret = []
        self.dfs("", left=0, right=0)
        return self.ret

if __name__ == "__main__":
    s = Solution()
    res = s.generateParenthesis(4)