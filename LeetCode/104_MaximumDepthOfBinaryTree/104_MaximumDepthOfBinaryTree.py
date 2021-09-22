# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 使用递归式，思路很简单。
class Solution(object):
    def maxDepth(self, root):
        if root == None:
            return 0
        leftNode = root.left
        rightNode = root.right
        leftDepth = self.maxDepth(leftNode)
        rightDepth = self.maxDepth(rightNode)
        return 1 + max(leftDepth, rightDepth)