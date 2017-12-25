# encoding=utf-8
from os import chdir
import nltk

chdir('C:\\nltk_data\\corpora\\conll2000')
train_sents = list(nltk.corpus.conll2000.iob_sents('train_preview.txt'))  # 多条训练语句及其对应的单词词性和单词标签
test_sents = list(nltk.corpus.conll2000.iob_sents('test_preview.txt'))  # 多条测试语句及其对应的单词词性和单词标签


def write_file(sentences, filename):
    with open(filename, 'w') as file:
        for sentence in sentences:
            for token, pos, label in sentence:
                file.write(''.join(['word=', token, ' pos=', pos, ' ', label, '\n']))
            file.write('\n')

write_file(train_sents, 'train_preview_prefix_added.txt')
write_file(test_sents, 'test_preview_prefix_added.txt')
