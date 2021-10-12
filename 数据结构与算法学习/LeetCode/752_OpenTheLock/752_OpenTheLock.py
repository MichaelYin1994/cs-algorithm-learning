# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:43:46 2019

@author: XPS13
"""

# collections的dequeue类，高效的双端队列的实现。
import collections
class Solution(object):
    def openLock(self, deadends, target):
        if target is None or target in deadends:
            return -1
        
        # Storing all deadends in a hash table
        deadends, visited, level = set(deadends), set(), 0
        queue = collections.deque()
        
        # Breath first search for the target
        queue.append("0000")
        
        while(len(queue) != 0):
            currSize, level = len(queue), level + 1
            
            for i in range(currSize):
                node = queue.popleft()
                
                # Early stop
                if node in deadends or node in visited:
                    continue
                
                # Current node possible adjacent nodes
                possibleLocks = []
                for i in range(4):
                    possibleLocks.append(node[:i] + str((int(node[i]) + 1) % 10) + node[(i+1):] )
                    possibleLocks.append(node[:i] + str((int(node[i]) + 9) % 10) + node[(i+1):] )
                
                # Travsel the possible nodes
                for j in possibleLocks:
                    if j == target:
                        return level
                    elif j not in deadends and j not in visited:
                        queue.append(j)
                visited.add(node)
        return -1

if __name__ == "__main__":
    deadends = ["0201","0101","0102","1212","2002"]
    target = "0202"
    
    deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"]
    target = "8888"
    s = Solution()
    ret = s.openLock(deadends, target)