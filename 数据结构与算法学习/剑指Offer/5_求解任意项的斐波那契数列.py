# 思路：借助Meomory这个外部数组，将Fibonacci数列的后面的项目给记住了
# 随后append到Fibonacci数列后面
class Solution:
    def Fibonacci(self, n):
        if n == 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        
        memory = [0, 1, 1]
        for i in range(3, n+1):
            tmp = memory[i-1] + memory[i-2]
            memory.append(tmp)
        return memory[-1]