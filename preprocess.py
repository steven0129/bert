import jieba
from tqdm import tqdm
from sklearn import preprocessing
from collections import Counter

jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)
le = preprocessing.LabelEncoder()

with open('pair.csv') as PAIR:
    le = preprocessing.LabelEncoder()
    texts = []
    texts.append('<UNK>')

    for line in tqdm(PAIR):
        [text, summary] = line.split(',')
        texts.append('<SOS>')
        texts.append('<EOS>')
        paras = text.split('<newline>')
        for para in paras:
            texts.extend(list(jieba.cut(para)))
            texts.append('<newline>')
        texts.extend(['<PAD>'] * ((510 - len(text)) + (510 - len(summary))))
        texts.extend(list(jieba.cut(summary)))
    
    counts = sorted(Counter(texts).items(), key=lambda x: x[1], reverse=True)
    sorted_texts = list(map(lambda x: x[0], counts))
    print(sorted_texts[:10])

    with open('bert-model/TF.csv', 'w') as TF:
        for text, count in tqdm(counts):
            TF.write(f'{text},{count}\n')