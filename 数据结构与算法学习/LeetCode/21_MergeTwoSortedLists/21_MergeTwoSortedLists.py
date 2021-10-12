# 思路很简单，使用虚拟的头结点保存地址
class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 == None:
            return l2
        elif l2 == None:
            return l1
        head_1 = l1
        head_2 = l2
        curr = ListNode(None)
        dummyNode = ListNode(None)
        dummyNode.next = curr
        while( head_1 != None and head_2 != None):
            if head_1.val >= head_2.val:
                curr.val = head_2.val
                tmp = ListNode(None)
                curr.next = tmp
                curr = curr.next
                head_2 = head_2.next
            else:
                curr.val = head_1.val
                tmp = ListNode(None)
                curr.next = tmp
                curr = curr.next
                head_1 = head_1.next
                
        if head_1 != None:
            while(True):
                if head_1 != None and head_1.next != None:
                    curr.val = head_1.val
                    tmp = ListNode(None)
                    curr.next = tmp
                    curr = curr.next
                    head_1 = head_1.next
                else:
                    curr.val = head_1.val
                    break
        else:
            while(True):
                if head_2 != None and head_2.next != None:
                    curr.val = head_2.val
                    tmp = ListNode(None)
                    curr.next = tmp
                    curr = curr.next
                    head_2 = head_2.next
                else:
                    curr.val = head_2.val
                    break
        return dummyNode.next