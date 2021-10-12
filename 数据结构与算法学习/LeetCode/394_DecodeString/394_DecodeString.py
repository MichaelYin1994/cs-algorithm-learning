# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:47:39 2019

@author: XPS13
"""

class Solution(object):
    def reverse_string(self, s):
        return s[::-1]
    
    def decodeString(self, s):
        if len(s) == 0:
            return ""
        
        # Initializing som params
        stack, ret = [], ""
        integer = set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
        
        # Scan the string
        for i in s:
            if i is not "]":
                stack.append(i)
            else:
                repeatChar, repeatInt, tmp = [], "", stack.pop()
                # Get chars
                while(tmp != "["):
                    repeatChar.append(tmp)
                    tmp = stack.pop()
                repeatChar.reverse()
                
                # Get the number of show times
                while(len(stack) != 0 and stack[-1] in integer):
                    repeatInt += stack.pop()
                repeatInt = int(self.reverse_string(repeatInt))
                
                # Push back to the stack
                stack.extend(repeatChar * repeatInt)
        
        # Construct the output
        for i in stack:
            ret += i
        return ret
if __name__ == "__main__":
    c = "3[z]2[2[y]pq4[2[jk]e1[f]]]ef"
    
    s = Solution()
    ret = s.decodeString(c)