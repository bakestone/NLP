import unicodedata
import re
import random
import torch
import jieba

device = torch.device("cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS andEOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?]+", r" ", s)
    return s

def tokenize_jieba(sentence, lang):
    if lang == 'zh':
        return ' '.join(jieba.cut(sentence))
    else:
        return sentence

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    lines = open('data2/reduce_%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    pairs = []
    for l in lines:
        splitted = l.split('\t')
        if len(splitted) != 2:  # 检查每一行是否包含两个句子
            #print(f"Skipping malformed line: {l}")
            continue
        pairs.append([normalizeString(tokenize_jieba(s, lang1 if i == 0 else lang2)) for i, s in enumerate(splitted)])
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

MAX_LENGTH = 200
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


from sklearn.model_selection import train_test_split


def prepareData(lang1, lang2, reverse=False, test_size=0.2, val_size=0.1):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    # 划分训练集和测试集
    train_pairs, test_pairs = train_test_split(pairs, test_size=test_size, random_state=42)

    # 划分训练集和验证集
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=val_size, random_state=42)

    return input_lang, output_lang, train_pairs, val_pairs, test_pairs


input_lang, output_lang, train_pairs, val_pairs, test_pairs = prepareData('en', 'zh', True)

# 打印一些样本
print("Training set samples:")
print(random.choice(train_pairs))

print("Validation set samples:")
print(random.choice(val_pairs))

print("Test set samples:")
print(random.choice(test_pairs))