# coding:utf-8
import pickle
import numpy as np
import jieba
import re


# texts = [
# [word,word, word,...],
#  ...
# ]
def wordFrequency(texts):
    dic = {}
    for words in texts:
        for word in words:
            try:
                dic[word] += 1
            except KeyError:
                dic[word] = 1
    return dic


# inputs a file and words are split by ' '(space)
def fileWordFrequency(path, useSpliter=False):
    dic = {}
    with open(path) as f:
        for l in f.readlines():
            if useSpliter:
                words = [i.encode('utf-8') for i in jieba.cut(l.strip())]
            else:
                words = l.strip().split(' ')
            for word in words:
                try:
                    dic[word] += 1
                except KeyError:
                    dic[word] = 1
    return dic


# sentences=[
#     [word, word,...]
#     [word, word, ...]
# ]
def indexText(sentences, dic, addDict=True):
    indexedText = []
    for sen in sentences:
        indexedSen, dic = indexSentence(sen, dic, addDict)
        indexedText.append(indexedSen)
    return indexedText, dic


# sentence = [ word, word, word ]
# 0: padding
# 1: unknown
# if unkonwn index is less than 1, than the unknon symbol would NOT be added into index sentence
def indexSentence(sentence, dic, addDict=False, unknownIndex=-1):
    count = dic.__len__() + 2
    indexSen = []
    try:
        for word in sentence:
            try:
                if word not in dic:
                    if addDict:
                        indexSen.append(count)
                        dic[word] = count
                        count += 1
                    else:
                        if unknownIndex > 0:
                            indexSen.append(unknownIndex)
                else:
                    indexSen.append(dic[word])
            except:
                print word
    except:
        print "sentence is not in right format"
    return indexSen, dic


# padding with 0
def padding(sentence, len):
    if sentence.__len__() == len:
        return sentence[:]
    elif sentence.__len__() < len:
        tmp = sentence[:]
        tmp.extend([0] * (len - sentence.__len__()))
        return tmp
    elif sentence.__len__() > len:
        return sentence[:len]


def reverseDic(dic):
    ret = {}
    try:
        for k, v in dic.items():
            ret[v] = k
    except:
        print 'reverse dict error'
    return ret


def id2String(data, dic):
    str = ""
    for i in data:
        if i == 0: break
        if i in dic:
            str += dic[i]
        else:
            str += '$UNC$'
    return str


def savePickle(obj, file):
    with open(file, 'w') as f:
        pickle.dump(obj, f)


def loadPickle(file):
    with open(file) as f:
        return pickle.load(f)

def saveDict(dic, file):
    with open(file, 'w') as f:
        pickle.dump(dic, f)


def loadDict(file):
    with open(file) as f:
        return pickle.load(f)


def saveItems(items, file, splitTag='\t'):
    with open(file, 'w') as fw:
        for i, v in items:
            try:
                try:
                    i = i.encode("utf-8")
                except:
                    pass
                fw.write(str(i) + splitTag + str(v) + '\n')
            except:
                print i
        fw.close()


def loadItems(file, splitTag='\t'):
    items = []
    with open(file) as f:
        for l in f.readlines():
            ele = l.strip().split(splitTag)
            items.append((ele[0], ele[1]))
        f.close()
    return items


def items2Dic(items):
    ret = {}
    for i, v in items:
        ret[i] = v
    return ret


def genRelationFromSentence(tagList, sentence, model=None):
    if model is None:
        model = {}
    for i in tagList:
        if i not in model:
            subdic = {}
            model[i] = subdic
        else:
            subdic = model[i]
        for w in sentence:
            try:
                subdic[w] += 1
            except:
                subdic[w] = 1
    return model


def sortDicByKeyAndReindex(dic, startIndex=0):
    its = dic.items()
    its = sorted(its, key=lambda x: x[0])
    count = startIndex
    sortedDic = {}
    for i in its:
        sortedDic[i[0]] = count
        count += 1
    return sortedDic


# 实现一个类似于bucket的padding操作, PS:会改变传入的实参
# maxPaddingLen表示最长的padding距离
def batchPadding(all, maxPaddingLen=100):
    for i in range(len(all)):
        maxBatchLen = min([max([len(k) for k in all[i]]), maxPaddingLen])
        for k in range(len(all[i])):
            all[i][k] = padding(all[i][k], maxBatchLen)


def oneHotALabel(label, maxLabelId, onValue=1.0, offValue=0.0):
    ret = [0] * (maxLabelId + 1)
    if type(label) is list:
        for i in label:
            ret[int(i)] = onValue
    else:
        ret[int(label)] = onValue
    return ret


def oneHotLabels(labels, maxLabelId=None, onValue=1.0, offValue=0.0):
    if maxLabelId is None:
        try:
            try:
                maxList = [max(k) for k in labels if len(k) > 0]
            except Exception as e:
                print e.message
            maxLabelId = max(maxList)
        except:
            maxLabelId = max(labels)
    ret = []
    for i in labels:
        ret.append(oneHotALabel(i, maxLabelId, onValue, offValue))
    return ret


def accuracy(predict, label):
    acc = 0
    predictLabel = np.argmax(predict, axis=1)
    for i, v in enumerate(predictLabel):
        if label[i][v] == 1:
            acc += 1.0
    acc /= len(predict)
    return acc


def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename, 'r')
    line = fr.readline().decode('utf-8').strip()
    # print line
    word_dim = int(line.split(' ')[1])
    vocab.append("unk")
    embd.append([0] * word_dim)
    vocab.append("padding")
    embd.append([0] * word_dim)
    for line in fr:
        row = line.strip().split(' ')
        if len(row) != word_dim + 1:
            continue
        vocab.append(row[0])
        embd.append([float(i) for i in row[1:]])
    print "loaded word2vec"
    fr.close()
    return vocab, embd


numReg = re.compile(u"\d+")
def replaceNum(line):
    try:
        line = line.decode('utf-8')
    except:
        pass
    return re.sub(numReg, u"num", line)


punctionReg = re.compile(
    u"[\[\]。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑•¨….¸;！´？\！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―\\\/\ˆ˜﹍﹎+=<\-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾\ˉ﹂﹄︼❝❞;]")
def replacePunctuation(line):
    try:
        line = line.decode('utf-8')
    except:
        pass
    t = re.sub(punctionReg, u'', line)
    return t


def indexCorpus(texts, minCount=1, removeNum=True, removePunc=True, needSplit=True):

    preprocessTexts = []
    for sen in texts:
        if removeNum:
            sen = replaceNum(sen)
        if removePunc:
            sen = replacePunctuation(sen)
        if needSplit:
            sen = [w for w in jieba.cut(sen)]
        preprocessTexts.append(sen)

    countDic = wordFrequency(preprocessTexts)
    tmp = {}
    indexCount = 2
    for i, v in countDic.items():
        if v >= minCount:
            tmp[i] = indexCount
            indexCount += 1
    indexDic = tmp
    X, indexDic = indexText(preprocessTexts,indexDic,addDict=False)
    return X, indexDic


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring