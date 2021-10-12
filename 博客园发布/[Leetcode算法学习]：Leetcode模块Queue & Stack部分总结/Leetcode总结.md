# Leetcode中的Queue & Stack部分学习总结
本月完成了Leetcode的Explore模块中额"Queue and Stack"部分。目前尚欠缺两道题：01 Matrix（DFS与DP结合，还有巧解法）与Clone Graph。今天小小的总结一下这一模块。该模块的目标在于：
1. 理解FIFO与LIFO的处理顺序。
2. 实现栈(Stack)与队列两种数据结构。
3. 对于Python或者其他语言，例如Java的内建栈与队列结构熟悉。
4. 运用Queue解决基本的BFS问题（重要）。
5. 运用Stack解决基本的DFS问题（重要）。

相关的题目包括：
| 题号 | 题目名 |  类型 |
| ---- | ---- | ----|
| 622 | Design Circular Queue | Queue |
| 200 | Number of Islands | DFS, Connected Components|
| 752 | Open the Lock| BFS, Queue |
| 279 | Prefect Squares | BFS, Queue, Dynamic Programming |
| 150 | Evaluate Reverse Polish Notation | Stack |
| 394 | Decode String | Stack |
| 155 | Min Stack | Stack |
| 20 | Valid Parentheses | Stack |
| 739 | Daily Temperatures | Stack |
| 494 | Target Sum | Stack, Dynamic Programming |
| 733 | Flood Fill | DFS |
| 542 | 01 Matrix | Dynamic Programming, DFS |
| 94 | Binary Tree Inorder Traversal | DFS, Stack |
| 133 | Clone Graph | Stack |
| 841 | Keys and Rooms | Stack DFS |
| ... | ... | ... |
---

## Queue与BFS部分

### Queue队列基本知识与重点关注点

对于Queue而言，该部分简要介绍了基本的Queue与Circular Queue。二者互有优劣，基本的Queue实现简单并且稳定，但是基本的Queue需要开辟定长的内存空间，当有元素出队之后，内存空置导致存储空间的浪费。循环队列可以优化队列的内存使用，循环队列保持了两个指针，heah指针指向队首，tail指针，当tail到尾部之后，可以存储于原来已经出队的元素的位置，而该位置的获取利用head来获得。

对于基本队列的题目Explore部分倒是没有多少，基本是将Queue嵌入到BFS中。但是一般Problems中Queue的题倒是挺多的，这里不再赘述。对于BFS而言，典型的问题为752 Open The Lock与279 Prefect Squares。对于BFS的模板而言，强烈建议参考这两题的范式。

以下为752 Open The Lock的BFS方法解。这里使用了Python原生collections中的dequeue类，也是双端队列高效的实现。该题意思是给定一个四位数的锁，每次可以改变锁的一位，例如8变到9，9变到0等，题目给定一个目标，随后计算从当前密码需要多少次变换变到目标密码。题目中给定了一些密码，当锁转到该密码时，锁会损坏，这说明在BFS的过程中，可以进行剪枝而缩小搜索空间。解题代码如下：
```python
import collections
class Solution(object):
    def openLock(self, deadends, target):
        if target is None or target in deadends:
            return -1
        
        # Storing all deadends in a hash table for searching efficiency
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
                
                # Ppossible adjacent nodes of the current node
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
                
                # Pruing, preventing the infinity loops
                visited.add(node)
        return -1
```

类似的题目还有279 Prefect Squares。不过该题存在DP的解法与BFS的解法。这里也使用了Python原生collections中的dequeue类。题意是给定某正整数n，计算构成n的最少的完美平方数的个数，所谓的完美平方数是指[1, 2, 4, 9...]。BFS的解法如下：
```python
import collections
class Solution(object):
    def numSquares(self, n):
        # Error input
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        
        # Generating the square numbers, and early stop
        squareNumbers, i = [], 1
        while(i * i <= n):
            if i * i == n:
                return 1
            else:
                squareNumbers.append(i * i)
                i += 1
                
        # Breath first search for the target                
        queue, level, visited = collections.deque(), -1, set()
        queue.append(n)
        while(len(queue) != 0):
            currSize, level = len(queue), level + 1
            for i in range(currSize):

                # Empty the queue in every iteration.
                node = queue.popleft()
                if node == 0:
                    return level
                for j in squareNumbers:
                    tmp = node - j

                    # Note that the visited array reduces the searching space of algorithm
                    if tmp >= 0 and tmp not in visited:
                        queue.append(tmp)
                        visited.add(tmp)
        return level
```

同时，279题还存在DP的解法，代码如下：
```python
class Solution(object):
    def numSquares(self, n):
        if n == 0:
            return None
        elif n == 1:
            return 1
        
        # Create the dp matrix
        dp = [i for i in range(0, n + 1)]
        
        # Opitmal condition:
        # dp[i](Prefect Squares) = dp[i - j*j] + 1 while j*j < i
        # Maximum Prefect Squares == i itself( 1 + 1 + ... + 1 == i)
        for i in range(1, n + 1):
            j = 1
            while(j * j <= i):
                dp[i] = min(dp[i], dp[i - j*j] + 1)
                j += 1
        return dp[-1]
```
DP的解法比较直观，对于数i而言，其需要的最多的完美平方数的个数为n，意即 1 + 1 + 1... + 1 = i ；因此，设dp[i]为组成数i所需要的完美平方数的个数，其值必定为dp[i - PrefectSquares]所需要的完美平方数加1。因此会给定变量j，随后遍历 j * j 所能够取到的完美平方数。

在该模块中，200 Number of Islands也作为BFS的经典题在模块里，但是由于Number of Islands更经常作为DFS经典题，因此不放在这里进行讨论。

---

## Stack部分

模块首先介绍了栈的基本知识，随后给出了DFS的Java两个模板。第一个模板为（递归DFS）：
```java
/*
 * Return true if there is a path from cur to target.
 */
boolean DFS(Node cur, Node target, Set<Node> visited) {
    return true if cur is target;
    for (next : each neighbor of cur) {
        if (next is not in visited) {
            add next to visted;
            return true if DFS(next, target, visited) == true;
        }
    }
    return false;
}
```
第二个模板为（非递归DFS）：
```java
/*
 * Return true if there is a path from cur to target.
 */
boolean DFS(int root, int target) {
    Set<Node> visited;
    Stack<Node> stack;
    add root to stack;
    while (s is not empty) {
        Node cur = the top element in stack;
        remove the cur from the stack;
        return true if cur is target;
        for (Node next : the neighbors of cur) {
            if (next is not in visited) {
                add next to visited;
                add next to stack;
            }
        }
    }
    return false;
}
```

由于之前笔者一直是学习的普林斯顿的《算法》那本书，书中的模板与上面的模板稍有不同，并且稍显冗余，因此觉得可以借鉴一下Leetcode所给出的DFS模板。对于Stack部分而言，明显可以看出存在两种类型的题目。第一种类型是经典的Stack的应用，另外一种是栈在DFS方面的应用。

对于栈的经典应用方面，以两道题为例进行分析。第一题是150 Evaluate Reverse Polish Notation，也就是著名的逆波兰表达式(RPN)的解析。逆波兰表达式是一种适合于计算机处理的字符表达方式，例如对于 "2 + 3" 而言，其逆波兰表达式为 "2 3 +" 。其题解如下：
```python
class Solution(object):
    def evalRPN(self, tokens):
        if len(tokens) == 1:
            return int(tokens[0])
        
        stack, ops = [], ["*", "+", "/", "-"]
        for c in tokens:
            if c not in ops:
                stack.append(int(c))
            else:
                rear = stack.pop()
                front = stack.pop()
                if c is "*":
                    stack.append(front * rear)
                elif c is "+":
                    stack.append(front + rear)
                elif c is "-":
                    stack.append(front - rear)
                else:
                    stack.append(int(front / rear))
        return int(stack[0])
```

对于150题而言题目较为简单，这里不赘述其思路。第二道经典的栈的题目为394 Decode String，题意为给定类似 "3[a]2[bc]" 的字符串，生成 "aaabcbc" 这样的字符串。一开始这道题死都过不了，因为笔者在字符串按字符出栈之后，将其组装成了一个大的字符串，随后又入栈。这导致在最后的反向的过程中，组装的字符串被整个翻转了一遍。正确的做法是按字符出栈，按字符反向，然后按字符入栈。以下为解题代码：
```python
class Solution(object):
    def reverse_string(self, s):
        return s[::-1]
    
    def decodeString(self, s):
        if len(s) == 0:
            return ""
        
        # Initializing some params
        stack, ret = [], ""
        integer = set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
        
        # Scan the string
        for i in s:
            if i is not "]":
                stack.append(i)
            else:
                repeatChar, repeatInt, tmp = [], "", stack.pop()
                # Get chars sequences
                while(tmp != "["):
                    repeatChar.append(tmp)
                    tmp = stack.pop()
                repeatChar.reverse()  # Reverse the char sequences
                
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
```

除了以上两道题之后，还有494 Target Sum这道让人头疼的题。朴素版本的DFS导致TLE，因此必须使用DP的思想来做这个问题。该题给定了一个数组num，要求计算num中的数字通过加减运算能够得到target的组合的个数。例如num = [1, 1, 1, 1, 1]，target为3，则返回5。代码如下：
```python
class Solution(object):
    def findTargetSumWays(self, nums, S):
        # Prevent the invalid input and the out-of-range sum results
        if len(nums) == 0:
            return None
        if S > sum(nums)  or S < - sum(nums):
            return 0
        
        # Including 0 elements, sum to 0
        numSum = sum(nums)
        rows, cols = len(nums) + 1, numSum * 2 + 1
        
        # Create a dp matrix
        dp = []
        for i in range(rows):
            dp.append([0] * cols)
        dp[0][numSum] = 1
        
        # Optimal condition
        for i in range(1, rows):
            for j in range(cols):
                left, right = j - nums[i-1], j + nums[i-1]
                if left - numSum >= - numSum:
                    dp[i][j] += dp[i-1][left]
                if right - numSum <= numSum:
                    dp[i][j] += dp[i-1][right]
                    
        return dp[-1][numSum + S]
```
该题是典型的DP问题，可与背包问题联动。最先想到应该是基于递归的DFS，但是其时间复杂度应该是O(2^n)指数阶，n是nums包含的元素的个数；基于DP的思想可以对问题进行求解，时间复杂度O(n * sum(nums))。可以采用二维DP的方式，DP思路与最优性条件为：
1. dp[i][j]为利用nums[:i]相加（包括nums[i]），和等于j的方法的次数。
2. 状态转移方程：dp[i][j] = dp[i-1][j - nums[i]] + dp[i-1][j + nums[i]]
3. j的取值范围为[-sum(nums), sum(nums)]，为了索引方便应该是[0, 2 * sum(nums)]
4. 初始条件：dp[0][0 + sum(nums)] == 0
5. 注意边界条件
6. return dp[-1][target + sum(nums)]

对于经典的DFS问题而言，以下代码可以作为DFS的新范式进行参考，代码较为简洁（毕竟头条写DFS模板时候被喷冗余的那一幕仍然历历在目= =）。下题为733 Flood Fill洪水填充问题，也是计算机视觉中的非常经典的算法。
```python
class Solution():
    def floodFill(self, image, sr, sc, newColor):
        if len(image) == 0:
            return image
        elif image[sr][sc] == newColor:
            return image
        
        self.fill(image, sr, sc, newColor, image[sr][sc])
        return image
    
    def fill(self, image, sr, sc, newColor, color):
        # 进入递归之前，先判断传入坐标合不合法，否则直接返回。
        if (sr < 0) or (sr > len(image) - 1) or (sc < 0) or (sc > len(image[0]) - 1) or (image[sr][sc] != color):
            return

        image[sr][sc] = newColor
        self.fill(image, sr-1, sc, newColor, color)
        self.fill(image, sc+1, sc, newColor, color)
        self.fill(image, sc, sr-1, newColor, color)
        self.fill(image, sc, sr+1, newColor, color)
        return
```

随后又是一道难题，542 01 Matrix，首先朴素的想法是DFS + DP的解法。该题待续，到现在还没太明白其解法。等彻底弄明白了再回来补这道题 = =。

```python
from collections import deque
class Solution():
    def updateMatrix(self, matrix):
        if len(matrix) == 0:
            return matrix
        
        # Initializing the searching directions, queue and dp matrix
        searchDir = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        numRows, numCols = len(matrix), len(matrix[0])
        queue = deque()
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    queue.append([i, j])
                else:
                    matrix[i][j] = float("inf")
        
        # Start the queue inserting process
        while(len(queue) != 0):
            pos = queue.pop()
            searchPos = [[pos[0] + tmp[0], pos[1] + tmp[1]] for tmp in searchDir]
            dist = matrix[pos[0]][pos[1]]
            for nextPos in searchPos:
                # If the distance of the neighboor points is smaller than the 
                # current dist + 1, then there is no need to continue.(Dijkstra)
                if nextPos[0] < 0 or nextPos[0] >= numRows or nextPos[1] < 0 or nextPos[1] >= numCols or dist + 1 > matrix[nextPos[0]][nextPos[1]]:
                    continue
                else:
                    matrix[nextPos[0]][nextPos[1]] = dist + 1
                    queue.append(nextPos)
        return matrix
```