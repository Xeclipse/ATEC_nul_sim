import gensim
from gensim.models import Word2Vec
#
# file = '/Users/nali/PycharmProjects/ATEC_nul_sim/ATEC/data/sen_cases_cut'
# with open(file) as f:
#     corpus = [l.strip().decode('utf-8').split(u' ') for l in f.readlines()]
#
#
# model = Word2Vec(sentences=corpus, sg=1, size=100, window=5, min_count=1, negative=7,
#                                         sample=0.001, hs=1, workers=3)
# model.save('/Users/nali/PycharmProjects/ATEC_nul_sim/ATEC/data/mode_save/w2v')


model = Word2Vec.load('/Users/nali/PycharmProjects/ATEC_nul_sim/ATEC/data/mode_save/w2v')
id= 'start'
while id!='exit':
    id = raw_input('word:')
    id = id.decode('utf-8')
    ans = model.most_similar(positive=[id],topn=10)
    for i in ans:
        print i[0],i[1]