# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 使用递归的方法
class Solution:
    def invertTree(self, root):
        # 返回条件是没有根节点
        if root == None:
            return
        self.invertTree(root.left)
        self.invertTree(root.right)
        # 将左孩子的值与右孩子的值对调
        tmp = root.left
        root.right = tmp
        root.left = tmp
        return root