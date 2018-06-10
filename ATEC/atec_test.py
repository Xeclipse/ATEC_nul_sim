#coding:utf-8
import time

from utils.nlp_zero import Tokenizer, Word_Finder, Template_Finder, Parser, XTrie
from utils import textProcess as tp

import pandas as pd

class F:
    def __iter__(self):
        with open('./data/sen_cases') as f:
            for l in f:
                yield l.strip().decode('utf-8')



def gen_tokenizer():
    word_finder = Word_Finder(min_proba=5e-5, min_pmi=2)
    word_finder.train(F())
    word_finder.find(F())
    tokenizer = word_finder.export_tokenizer()
    tp.savePickle(tokenizer, '../ATEC/tokenizer')
    with open('./data/sen_cases_cut','w') as fout:
        with open('./data/sen_cases') as f:
            for l in f.readlines():
                res = tokenizer.tokenize(l.strip().decode('utf-8'))
                fout.write(u' '.join(res).encode('utf-8')+'\n')

def gen_templates():
    tokenizer = tp.loadPickle('../ATEC/tokenizer')
    f = Template_Finder(tokenizer.tokenize, window=3)
    f.train(F())
    f.find(F())
    templates = f.templates
    templates = {i:
        j for i,j in templates.items() if not i.is_trivial()
                 }
    trie = XTrie()
    for i,j in templates.items():
        trie[tuple(i.words)] = j
    tp.savePickle(trie, '../ATEC/templates')

if __name__ == '__main__':
    # gen_templates()
    tokenizer = tp.loadPickle('../ATEC/tokenizer')
    templates =tp.loadPickle('../ATEC/templates')
    start = time.time()
    p = Parser(templates, tokenizer.tokenize)
    tree1 = p.parse(u'我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号').plot()
    tree2 = p.parse(u'怎么更改花呗手机号码').plot()

    end = time.time()
    print tree1
    print tree2
    print (end - start)


# tp.savePickle(templates, './data/templates')