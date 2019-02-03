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

print('前處理詞頻...')

with open('pair.csv') as PAIR:
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

print('前處理FastText...')

with open('pair.csv') as PAIR:
    for line in tqdm(PAIR):
        [text, summary] = line.split(',')

        with open('bert-model/sents.txt', 'a') as OUT:
            OUT.write('<SOS> ')
            
            paras = text.strip().split('<newline>')
            for para in paras:
                for word in jieba.cut(para):
                    OUT.write(word + ' ')

                if len(paras) != 1: OUT.write('<newline> ')

            OUT.write('<EOS>\n')
            OUT.write('<SOS> ')
            for word in jieba.cut(summary.strip()):
                OUT.write(word + ' ')

            OUT.write('<EOS> ')

            for i in range(len(text) - len(summary)):
                OUT.write('<PAD> ')

            OUT.write('\n')
