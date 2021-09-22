# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:39:33 2019

@author: XPS13
"""
###############################################################################
class TrieNode:
    def __init__(self):
        self.isWord = False
        self.children = {}
        
###############################################################################
class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = TrieNode()
            node = node.children[i]
        node.isWord = True

    def search(self, word):
        node = self.root
        for i in word:
            if i not in node.children:
                return False
            node = node.children[i]
        return node.isWord

    def startsWith(self, prefix):
        node = self.root
        for i in prefix:
            if i not in node.children:
                return False
            node = node.children[i]
        
        self.startsWithWords = []
        self.in_order_traversal(node, prefix=prefix)
        return self.startsWithWords
    
    def in_order_traversal(self, node, prefix=""):
        if node is None:
            return prefix
        elif node.isWord is True:
            self.startsWithWords.append(prefix)
        
        for c in list(node.children.keys()):
            self.in_order_traversal(node.children[c], prefix=prefix+c)
            
###############################################################################
if __name__ == "__main__":
    words = ["Trie","insert","search","search","startsWith","insert","search"]
    
    trieTree = Trie()
    for word in words:
        trieTree.insert(word)
    
    print(trieTree.startsWith("in"))
    print(trieTree.startsWith("s"))
    print(trieTree.startsWith("se"))
