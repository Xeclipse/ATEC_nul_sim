# coding:utf-8
from collections import Counter

import jieba
import utils.textProcess as tp

jieba.add_word("花呗", freq=100000)
jieba.add_word("借呗", freq=100000)
jieba.add_word("外卖", freq=100000)
jieba.add_word("闲鱼", freq=100000)


csv_file = './data/atec_nlp_sim_train.csv'
word_dic_file = './data/processed_data/word_dic.dic'
corpus_save_file = './data/processed_data/corpus'

class Pair:
    def __init__(self):
        self.id = -1
        self.first_sen = None
        self.second_sen = None
        self.label = 0

    def __init__(self, id, sen1, sen2, label):
        self.id = id
        self.first_sen = sen1
        self.second_sen = sen2
        self.label = label

    def cut_word(self):
        self.cut_first_sen = [i for i in jieba.cut(self.first_sen)]
        self.cut_second_sen = [i for i in jieba.cut(self.second_sen)]
        return self.cut_first_sen, self.cut_second_sen

    def __init__(self, line):
        line = line.strip()
        self.id, self.first_sen, self.second_sen, self.label = line.split('\t')

    def filter_by_freq(self, freq_dic):
        self.cut_first_sen_filtered = [i for i in self.cut_first_sen if freq_dic[i] > 5]
        self.cut_second_sen_filtered = [i for i in self.cut_second_sen if freq_dic[i] > 5]
        return self.cut_first_sen_filtered, self.cut_second_sen

    def index(self, index_dic):
        self.index_first_sen = tp.indexSentence(sentence=self.cut_first_sen, dic=index_dic, addDict=False,
                                                unknownIndex=1)[0]
        self.index_second_sen = tp.indexSentence(sentence=self.cut_second_sen, dic=index_dic, addDict=False,
                                                 unknownIndex=1)[0]
        return self.index_first_sen, self.index_second_sen

    def remove_same_word(self):
        same = [i for i in self.cut_first_sen_filtered if i in self.cut_second_sen_filtered]
        self.remove_same_first_sen =[]
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

    def char_spilt(self):
        pass
    def __str__(self):
        ret =''
        ret += str(self.id)+'\n'
        ret += self.first_sen+'\n'
        ret += self.second_sen+'\n'
        ret += u' '.join(self.cut_first_sen).encode('utf-8')+'\n'
        ret += u' '.join(self.cut_second_sen).encode('utf-8')+'\n'
        ret += u' '.join(self.remove_same_first_sen).encode('utf-8') + '\n'
        ret += u' '.join(self.remove_same_second_sen).encode('utf-8') + '\n'
        ret += str(self.label)+'\n'
        ret +='-'*20+'\n'
        return  ret


def save_freq_dic(freq_sta):
    freq_items = sorted(freq_sta.items(), key=lambda x: x[1], reverse=True)
    with open('./word_statistic.txt', 'w') as f:
        for i, v in freq_items:
            f.write(i.encode('utf-8') + '\t' + str(v) + '\n')

def preprocess():
    corpus = []
    with open(csv_file) as f:
        corpus = [Pair(line) for line in f.readlines()]
    freq_sta = Counter()
    for pair in corpus:
        pair.cut_word()
        freq_sta.update(pair.cut_first_sen)
        freq_sta.update(pair.cut_second_sen)
    index_dic_items = [i for i in freq_sta.items() if i[1] > 5]
    index_dic = tp.items2Dic(index_dic_items)
    index_dic = tp.sortDicByKeyAndReindex(index_dic, startIndex=2)

    for pair in corpus:
        pair.filter_by_freq(freq_sta)
        pair.index(index_dic)
        pair.padding(padding_len=25)
        pair.remove_same_word()
    tp.saveDict(index_dic, word_dic_file)
    tp.savePickle(corpus,corpus_save_file)

# preprocess()
# corpus = tp.loadPickle(corpus_save_file)
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