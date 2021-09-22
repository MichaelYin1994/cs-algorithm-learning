# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:25:06 2019

@author: XPS13
"""
class Solution(object):
    def invalidTransactions(self, transactions):
        if len(transactions) == 0:
            return []
        
        # Transactions dict(by name)
        trans, invalidTrans = {}, []
        for item in transactions:
            currTrans = item.split(",")
            currTrans = [currTrans[0], int(currTrans[1]), int(currTrans[2]),
                         currTrans[3], item]
            if currTrans[0] not in trans:
                trans[currTrans[0]] = [currTrans]
                if currTrans[2] > 1000:
                    invalidTrans.append(item)
            else:
                for i in range(len(trans[currTrans[0]])):
                    prevTrans = trans[currTrans[0]][i]
                    if abs(currTrans[1] - prevTrans[1]) < 60 and (currTrans[3] != prevTrans[3]):
                        invalidTrans.append(item)
                        invalidTrans.append(prevTrans[-1])
                    elif currTrans[2] > 1000:
                        invalidTrans.append(item)
                trans[currTrans[0]].append(currTrans)
        invalidTrans = list(set(invalidTrans))
        return invalidTrans
    
if __name__ == "__main__":
    s = Solution()
#    transactions = ["alice,20,800,mtv","alice,50,100,beijing"]
#    transactions = ["alice,20,800,mtv","bob,50,1200,mtv"]
    transactions = ["alice,20,800,mtv","bob,50,1200,mtv"]
#    transactions = ["alice,20,800,mtv","alice,50,200,mtv", "alice,170,1200,beijing"]
    ret = s.invalidTransactions(transactions)