import jieba
from tqdm import tqdm
from collections import Counter

jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

hashTable = {}

def addKey(key):
    try:
        hashTable[key] += 1
    except:
        hashTable[key] = 1

with open('pair-lcstcs100000+origin.csv') as PAIR:
    addKey('<UNK>')

    for line in tqdm(PAIR):
        [text, summary] = line.split(',')
        addKey('<SOS>')
        addKey('<EOS>')
        paras = text.split('<newline>')
        for para in paras:
            for word in jieba.cut(para):
                addKey(word)
            addKey('<newline>')

        for i in range(len(text) - len(summary)):
            addKey('<PAD>')

        for word in jieba.cut(summary):
            addKey(word)
    
    counts = sorted(hashTable.items(), key=lambda x: x[1], reverse=True)
    top10_texts = list(map(lambda x: x[0], counts[:10]))
    print(f'Top 10: {",".join(top10_texts)}')

    with open('bert-model/TF.csv', 'w') as TF:
        for text, count in tqdm(counts):
            TF.write(f'{text},{count}\n')
