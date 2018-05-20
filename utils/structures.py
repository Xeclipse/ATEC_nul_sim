# coding=utf-8


class TrieTreeNode:
    def __init__(self):
        self.children = {}
        self.value = None
        self.count = 0

    def add_child(self,child_key, value = None):
        self.children[child_key] = TrieTreeNode(value)
        self.children[child_key].count += 1



class TrieTree:
    def __init__(self):
        self.root =TrieTreeNode()

    def add_seq(self, seq, start =0, end = None):
        if end is None:
            end = len(seq)
        node = self.root
        for i in range(start, end):
            key = seq[i]
            if key not in node.children:
                node.children[key] = TrieTreeNode()
            node = node.children[key]
            node.count += 1
    def add_seq_suffix(self, seq):
        for start in range(0, len(seq)):
            self.add_seq(seq, start = start, end = len(seq))

    # 如果找不到就返回最长的匹配长度
    def find_seq_max_len(self, seq, start = 0, end=None):
        if end is None:
            end = len(seq)
        if len(seq) ==0:
            return 0
        node = self.root
        for i in range(start, end):
            key = seq[i]
            if key not in node.children:
                return i-start
            else:
                node = node.children[key]
        return end - start



    def find_seq(self, seq, start = 0, end=None):
        if end is None:
            end = len(seq)
        if len(seq) ==0:
            return None
        node = self.root
        for i in range(start, end):
            key = seq[i]
            if key not in node.children:
                return None
            else:
                node = node.children[key]
        return node

    def find_coincidences(self, seq):
        end = len(seq)
        ret = []
        start_index = 0
        while start_index<end:
            found_len =  self.find_seq_max_len(seq,start_index, end)
            if found_len == 0:
                start_index += 1
            else:
                ret.append(seq[start_index:start_index+found_len])
                start_index+= found_len
        return ret

if __name__ == '__main__':
    trie = TrieTree()
    trie.add_seq_suffix(u"我很喜欢这个挑战")
    print u' '.join(trie.find_coincidences(u'这个小挑战'))