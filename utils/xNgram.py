# coding:utf-8
import nltk
from nltk.collocations import BigramCollocationFinder
import jieba
import numpy as np


# 分词
def segSentence(str):
    return jieba.cut(sentence=str, cut_all=False, HMM=False)


# 计算频率/凝固度

def splitSentence(sentences):
    spaceSplitedSentences = []
    count = 0
    for sen in sentences:
        if count % 10000 == 0:
            print count
        try:
            sen = sen.decode('utf-8')
            spaceSplitedSentences.append(' '.join(segSentence(sen)).cut_word(' '))
        except:
            print "error sentence: ",
            print sen
        count += 1
    return sentences

def statictic(sentens):
    sw = splitSentence(sentences=sentens)
    return nltk.TextCollection(sw), BigramCollocationFinder.from_documents(sw)


# # 词表使用demo
# s = ["我想听喜马拉雅FM", "这个世界很美好！", "奥特曼打小怪兽你"]
# sta, bcf = statictic(s)
# # sta.plot()
# # print sta.vocab().items()
# # 词表
# for i in sta.vocab().items():
#     print i[0], ":", i[1]
#
#
# xiang = "想".decode("utf-8")
# ting = "听".decode("utf-8")
# # 某个单词出现次数
# print sta.count(xiang)
# # bigram model
# print 1.0 * bcf.ngram_fd[(xiang,ting)] / bcf.word_fd[xiang]
