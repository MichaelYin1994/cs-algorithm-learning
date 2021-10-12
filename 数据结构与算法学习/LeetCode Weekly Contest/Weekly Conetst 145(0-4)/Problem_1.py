# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:18:47 2019

@author: XPS13
"""

class Solution(object):
    def relativeSortArray(self, arr1, arr2):
        arr1_count, retArr = {}, []
        for i in arr1:
            if i not in arr1_count:
                arr1_count[i] = 1
            else:
                arr1_count[i] += 1
        arr2_count = set(arr2)
        sortedKey = sorted(list(arr1_count.keys()))
        
        for i in range(len(arr2) - 1):
            num = arr2[i]
            retArr.extend(arr1_count[num] * [num])
            del arr1_count[num]
            for j in sortedKey:
                if (j >= num) and (j < arr2[i+1]) and (j not in arr2_count):
                        retArr.extend(arr1_count[j] * [j])
                        del arr1_count[j]
                        
        retArr.extend(arr1_count[arr2[-1]] * [arr2[-1]])
        del arr1_count[arr2[-1]]
        
        for i in sortedKey:
            if i in arr1_count.keys():
                retArr.extend(arr1_count[i] * [i])
        
        return retArr
    
if __name__ == "__main__":
    s = Solution()
    arr1, arr2 = [2,3,1,3,2,4,6,7,9,2,19], [2,1,4,3,9,6]
    
    ret = s.relativeSortArray(arr1, arr2)
    ans = [2,2,2,1,4,3,3,9,6,7,19]
    print(ret)
    