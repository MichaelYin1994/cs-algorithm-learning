# 使用字典哈希，easy的难度。但是特别注意另外一种解法
# 因为数组已经排序，可以考虑左右夹逼，得到结果（快排思想）
class Solution:
    def FindNumbersWithSum(self, array, target):
        if len(array) == 0:
            return []
        memory = {}
        member = {}
        for ind, item in enumerate(array):
            s = target - item
            if item not in memory.keys():
                memory[s] = ind
            else:
                member[item * s] = [item, s] if item < s else [s, item]
        if len(member) == 0:
            return []
        elif len(member) == 1:
            ind = list(member.keys())[0]
            return member[ind]
        else:
            ind = min(list(member.keys()))
            return member[ind]