# 存在两种思路，一种是冒泡法，一直让新的奇数往下沉，时间复杂度O(n^2)，空间复杂度O(1)
# 另外一种是归排的思想，用空间换时间，确立头指针和尾指针
# 建立两个辅助数组，一个找奇数，一个找偶数，最后连起来
# 时间复杂度O(n)，空间复杂度O(n)

class Solution:
    def reOrderArray(self, array):
        for i in range(len(array)):
            if array[i] % 2 == 1:
                j = i - 1
                while( j >= 0 ):
                    if array[j] % 2 == 0:
                        tmp = array[j]
                        array[j] = array[j + 1]
                        array[j + 1] = tmp
                        j -= 1
                    else:
                        j = -1
        return array