# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:12:12 2018

@author: Administrator
"""

class Solution:
    def union(self, pRoot, qRoot):
        if self.treeSize[pRoot] >= self.treeSize[qRoot]:
            self.id[qRoot] = self.id[pRoot]
            self.treeSize[pRoot] += self.treeSize[qRoot]
        else:
            self.id[pRoot] = self.id[qRoot]
            self.treeSize[qRoot] += self.treeSize[pRoot]
            
    def find(self, p):
        while (self.id[p] != p):
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p
    
    def findCircleNum(self, M):
        if len(M) == 0:
            return 0
        peopleNums = len(M)
        self.id = list(range(peopleNums))
        self.treeSize = [1] * peopleNums
        
        for p, item_rol in enumerate(M):
            for q, item_col in enumerate(item_rol):
                if item_col == 1:
                    pRoot = self.find(p)
                    qRoot = self.find(q)
                    if pRoot == qRoot:
                        continue
                    else:
                        self.union(pRoot, qRoot)
                        
        friendCicle = 0
        for ind, item in enumerate(self.id):
            if ind == item:
                friendCicle += 1
        return friendCicle
    
if __name__ == "__main__":
    M_1 = [[1,1,0],
         [1,1,0],
         [0,0,1]]
    M_2 = [[1,1,0],
         [1,1,1],
         [0,1,1]]
    s = Solution()
    print(s.findCircleNum(M_2))