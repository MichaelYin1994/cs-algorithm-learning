# 动态规划典型问题，参见：http://www.csie.ntnu.edu.tw/~u91029/DynamicProgramming.html#1
class Solution:
    def jumpFloor(self, number):
        if number == 1:
            return 1
        elif number == 2:
            return 2
        # 要增加一个虚拟解值0，否则数组索引错误
        memory = [0, 1, 2]
        for i in range(3,number+1):
            memory.append(memory[i-1] + memory[i-2])
        return memory[-1]
        