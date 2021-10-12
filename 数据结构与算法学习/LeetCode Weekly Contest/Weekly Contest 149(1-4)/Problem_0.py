# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:25:06 2019

@author: XPS13
"""
class Solution(object):
    def dayOfYear(self, date):
        if len(date) == 0:
            return None
        monthBig = [1, 3, 5, 7, 8, 10, 12]
        date = list(map(int, date.split("-")))
        if ((date[0] % 4 == 0 and date[0] % 100 != 0) or (date[0] % 400 == 0)):
            yearFlag = 1
        else:
            yearFlag = 0
        
        days = 0
        for month in range(1, date[1]):
            if month == 2 and yearFlag == 1:
                days = days + 29
            elif month == 2 and yearFlag == 0:
                days = days + 28
            elif month in monthBig:
                days = days + 31
            else:
                days = days + 30
        days = date[2] + days
        return days
        

if __name__ == "__main__":
    s = Solution()
    ret = s.dayOfYear("2019-01-09")