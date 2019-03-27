from gensim.models import word2vec
from sklearn.decomposition import PCA
from gensim.models.fasttext import FastText
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

model = FastText.load_fasttext_format('bert-model/wordvec-large.bin', full_model=False)
raw_word_vec = model.wv.vectors

cent_word1 = "段譽"
cent_word2 = "蕭峰"
cent_word3 = "虛竹"

wordList1 = model.wv.most_similar_cosmul(cent_word1)
wordList2 = model.wv.most_similar_cosmul(cent_word2)
wordList3 = model.wv.most_similar_cosmul(cent_word3)


wordList1 = np.append([item[0] for item in wordList1], cent_word1)
wordList2 = np.append([item[0] for item in wordList2], cent_word2)
wordList3 = np.append([item[0] for item in wordList3], cent_word3)

def get_word_index(word):
    index = model.wv.vocab[word].index
    return index

index_list1 = map(get_word_index, wordList1)
index_list2 = map(get_word_index, wordList2)
index_list3 = map(get_word_index, wordList3)

vec_reduced = PCA(n_components=2).fit_transform(raw_word_vec)
# zhfont = matplotlib.font_manager.FontProperties(fname=r'C:\Nuance\python_env\basic_dev\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\msyh.ttf')
x = np.arange(-10, 10, 0.1)
y = x
plt.plot(x, y)

for i in index_list1:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='r', fontproperties=zhfont)

for i in index_list2:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='b', fontproperties=zhfont)

for i in index_list3:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='g', fontproperties=zhfont)

plt.savefig('vis.png')