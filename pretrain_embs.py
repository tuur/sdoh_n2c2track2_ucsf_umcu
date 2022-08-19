
import argparse, spacy
from brat_scoring.corpus import Corpus
from csv import reader
from gensim.models import FastText
from gensim.models import Word2Vec
from gensim import utils
from collections import Counter
import re, os

mimic_noteevents_csv_path="/Users/aleeuw15/Desktop/Research/NLP - TextImp/datasets/mimic/NOTEEVENTS.csv"

# 1 -------- tokenization
nlp = spacy.load("en_core_web_sm", exclude=["tagger","parser"])
class Itertexts:
    def __iter__(self):
        with utils.open(mimic_noteevents_csv_path, 'r', encoding='utf-8') as read_obj:
            for row in reader(read_obj):
                txt = row[10]
                #print(row[6])
                yield txt

print('get texts')
mimic_texts = list(Itertexts())
print('tokenize')
tokenized_mimic_texts = nlp.tokenizer.pipe(mimic_texts) #[ttext for ttext in ]
print('write out')
with open('mimic_tokenized.txt', 'w') as f:
    for ttext in tokenized_mimic_texts:
        f.write(str(ttext) + '\n')

# 2 ---------- introduce UNKs for words with freq 1

with open('mimic_tokenized.txt', 'r') as f:
    mimic_tokenized = f.read()
    tokens = re.split(',|\n', mimic_tokenized)
    print('num tokens',len(tokens))
    #tokens = mimic_tokenized.split(' \n')
    counts = Counter(tokens)


with open('mimic_tokenized_unk1.txt', 'w') as fout:
    with open('mimic_tokenized.txt', 'r') as fin:
        for line in fin.readlines():
            tokens = line.strip().split(' ')
            newtokens = [t if counts[t]> 1 else '<UNK>' for t in tokens]
            fout.write(' '.join(newtokens) + '\n')

# 3 ------ train the embeddings
# print('training embs')
# word2vec_command='python -m gensim.scripts.word2vec_standalone -train mimic_tokenized_unk1.txt -output mimic_tokenized_unk1_word2vec_5it_250.bin -size 250 -sample 1e-4 -cbow 0 -binary 0 -iter 5 -window 4'
# os.popen(word2vec_command)

fasttext_command = '"/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/code/N2C2-TR2-SOCDET/fastText/fasttext" skipgram -input "/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/code/N2C2-TR2-SOCDET/n2c2-tr2-socdet/mimic_tokenized_unk1.txt" -output "/Users/aleeuw15/Desktop/Research/N2C2 - SDOH/code/N2C2-TR2-SOCDET/n2c2-tr2-socdet/mimic_tokenized_unk1_fasttext_5it_250.bin" -ws 4 -epoch 5 -dim 250'
os.popen(fasttext_command)

