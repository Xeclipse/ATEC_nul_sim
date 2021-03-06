# coding:utf-8
from collections import Counter
from utils.structures import TrieTree, TrieTreeNode
import utils.textProcess as tp
from utils.nlp_zero import Tokenizer
# jieba.add_word("花呗", freq=100000)
# jieba.add_word("借呗", freq=100000)
# jieba.add_word("外卖", freq=100000)
# jieba.add_word("闲鱼", freq=100000)
# jieba.add_word("更改", freq=100000)

csv_file = './data/atec_nlp_sim_train.csv'
csv_file2 = './data/atec_nlp_sim_train_add.csv'
word_dic_file = './data/processed_data/word_dic.dic'
char_dic_file = './data/processed_data/char_dic.dic'
corpus_save_file = './data/processed_data/corpus'
tokenizer= tp.loadPickle('./tokenizer')

class Pair:
    def __init__(self):
        self.id = -1
        self.first_sen = None
        self.second_sen = None
        self.label = 0

    def __init__(self, id, sen1, sen2, label):
        self.id = id
        self.first_sen = sen1.replace('***', '&')
        self.second_sen = sen2.replace('***', '&')
        self.label = label

    def cut_word(self):
        self.cut_first_sen = [i for i in tokenizer.tokenize(self.first_sen)]
        self.cut_second_sen = [i for i in tokenizer.tokenize(self.second_sen)]
        return self.cut_first_sen, self.cut_second_sen

    def __init__(self, line):
        line = line.strip()
        self.id, self.first_sen, self.second_sen, self.label = line.replace('***', '&').split('\t')

    def filter_by_freq(self, freq_dic):
        self.cut_first_sen_filtered = [i for i in self.cut_first_sen if freq_dic[i] > 5]
        self.cut_second_sen_filtered = [i for i in self.cut_second_sen if freq_dic[i] > 5]
        return self.cut_first_sen_filtered, self.cut_second_sen

    def index(self, index_dic):
        self.index_first_sen = tp.indexSentence(sentence=self.cut_first_sen, dic=index_dic, addDict=False,
                                                unknownIndex=1)[0]
        self.index_second_sen = tp.indexSentence(sentence=self.cut_second_sen, dic=index_dic, addDict=False,
                                                 unknownIndex=1)[0]
        self.index_same_part = tp.indexSentence(sentence=self.same_part, dic=index_dic, addDict=False,
                                                unknownIndex=1)[0]
        self.index_diff_part_1 = tp.indexSentence(sentence=self.first_diff_part, dic=index_dic, addDict=False,
                                                  unknownIndex=1)[0]
        self.index_diff_part_2 = tp.indexSentence(sentence=self.second_diff_part, dic=index_dic, addDict=False,
                                                  unknownIndex=1)[0]
        return self.index_first_sen, self.index_second_sen, self.index_same_part, self.index_diff_part_1, self.index_diff_part_2

    def same_parts(self):
        self.same_part = []
        for i in self.cut_first_sen:
            if i in self.cut_second_sen:
                self.same_part.append(i)
        return self.same_part

    def diff_parts(self):
        self.same_parts()
        self.first_diff_part = []
        self.second_diff_part = []
        for i in self.cut_first_sen:
            if i not in self.same_part:
                self.first_diff_part.append(i)
        for i in self.cut_second_sen:
            if i not in self.same_part:
                self.second_diff_part.append(i)
        return self.first_diff_part, self.second_diff_part

    def remove_same_word(self):
        same = [i for i in self.cut_first_sen_filtered if i in self.cut_second_sen_filtered]
        self.remove_same_first_sen = []
        for i in self.cut_first_sen_filtered:
            if i in same:
                self.remove_same_first_sen.append(u'holder')
            else:
                self.remove_same_first_sen.append(i)

        self.remove_same_second_sen = []
        for i in self.cut_second_sen_filtered:
            if i in same:
                self.remove_same_second_sen.append(u'holder')
            else:
                self.remove_same_second_sen.append(i)
        return self.remove_same_first_sen, self.remove_same_second_sen

    def padding(self, padding_len):
        self.padding_index_first_sen = tp.padding(self.index_first_sen, padding_len)
        self.padding_index_second_sen = tp.padding(self.index_second_sen, padding_len)
        return self.padding_index_first_sen, self.padding_index_second_sen

    def __str__(self):
        ret = ''
        ret += str(self.id) + '\n'
        ret += self.first_sen + '\n'
        ret += self.second_sen + '\n'
        ret += u' '.join(self.cut_first_sen).encode('utf-8') + '\n'
        ret += u' '.join(self.cut_second_sen).encode('utf-8') + '\n'
        # ret +='coincident_parts: '+u' '.join(self.coincidence_parts).encode('utf-8') + '\n'
        ret += str(self.label) + '\n'
        ret += '-' * 20 + '\n'
        return ret

    def pinyin(self):
        pass

    def format_sen(self, sen):
        return sen.replace(r'\\d+', '$').replace('***', '&').decode('utf-8')

    def find_coincident_parts(self):
        first_sen_trie = TrieTree()
        first_sen_trie.add_seq_suffix(self.format_sen(self.first_sen))
        self.coincidence_parts = first_sen_trie.find_coincidences(self.format_sen(self.second_sen))
        return self.coincidence_parts

    def gen_pinyin(self):
        self.first_sen_pinyin = []
        self.second_sen_pinyin = []


def save_freq_dic(freq_sta):
    freq_items = sorted(freq_sta.items(), key=lambda x: x[1], reverse=True)
    with open('./word_statistic.txt', 'w') as f:
        for i, v in freq_items:
            f.write(i.encode('utf-8') + '\t' + str(v) + '\n')


def preprocess():
    dic_file = word_dic_file
    corpus = []
    with open(csv_file) as f:
        corpus = [Pair(line) for line in f.readlines()]
    with open(csv_file2) as f:
        corpus.extend([Pair(line) for line in f.readlines()])
    freq_sta = Counter()
    for pair in corpus:
        pair.cut_word()
        freq_sta.update(pair.cut_first_sen)
        freq_sta.update(pair.cut_second_sen)
        pair.diff_parts()
    # freq_sta = tp.loadDict(char_dic_file)
    index_dic_items = [i for i in freq_sta.items() if i[1] > 10]
    index_dic = tp.items2Dic(index_dic_items)
    index_dic = tp.sortDicByKeyAndReindex(index_dic, startIndex=2)

    for pair in corpus:
        # pair.filter_by_freq(freq_sta)
        pair.index(index_dic)
        pair.padding(padding_len=30)
        # pair.remove_same_word()
    tp.saveDict(index_dic, dic_file)
    tp.savePickle(corpus, corpus_save_file)
    print dic_file


def gen_coincidence_corpus():
    corpus = []
    with open(csv_file) as f:
        corpus = [Pair(line) for line in f.readlines()]
        for pair in corpus:
            pair.find_coincident_parts()
    with open('./conidence_parts', 'w') as f:
        for pair in corpus:
            f.write(pair.__str__() + '\n')


def output_pari(file):
    corpus = tp.loadPickle(corpus_save_file)
    with open(file, 'w') as f:
        for pair in corpus:
            f.write(pair.first_sen + '\n')
            f.write(pair.second_sen + '\n')


def output_neg_pos():
    pairs = tp.loadPickle(corpus_save_file)
    f1 = open('./pos_case', 'w')
    f2 = open('./neg_case', 'w')
    for pair in pairs:
        if pair.label == '1':
            f1.write(pair.__str__() + '\n')
        else:
            f2.write(pair.__str__() + '\n')
    f1.close()
    f2.close()

if __name__ == '__main__':


    preprocess()
    output_pari('./cases.txt')

    # suffix_tree = TrieTree()
    # gen_coincidence_corpus()
    # preprocess()
    #     corpus = tp.loadPickle(corpus_save_file)
    # debug = 0
    #
    # with open('./cases.txt', 'w') as f:
    #         for pair in corpus:
    #             f.write(str(pair))

    # with open('./word_filtered.txt', 'w') as f:
    #     for pair in corpus:
    #         s1, s2 = pair.filter_by_freq(freq_sta)
    #         s1, s2= pair.index(index_dic)
    # f.write(u' '.join(s1).encode('utf-8'))
    # f.write('\t')
    # f.write(u' '.join(s2).encode('utf-8'))
    # f.write('\n')
    # preprocess()
    # len_counter = Counter()
    # corpus = tp.loadPickle('/Users/nali/PycharmProjects/MachineLearningLaboratory/ATEC/data/processed_data/corpus')
    # for pair in corpus:
    #     len_counter.update([len(pair.index_first_sen),len(pair.index_second_sen)])
    # items = sorted(len_counter.items(), key=lambda x:x[1], reverse=True)
    # for i in items:
    #     print i[0],'\t',i[1]

    # data=[]
    # for pair in corpus:
    #     data.append(pair.first_sen)
    #     data.append(pair.second_sen)
    # char_dic = tp.wordFrequency(data)
    # tp.saveDict(char_dic,char_dic_file)
    # print char_dic
