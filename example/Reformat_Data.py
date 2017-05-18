from os import chdir
import nltk

chdir('C:\\nltk_data\\corpora\\conll2000')
train_sents = list(nltk.corpus.conll2000.iob_sents('train.txt'))  # 多条训练语句及其对应的单词词性和单词标签
test_sents = list(nltk.corpus.conll2000.iob_sents('test.txt'))  # 多条测试语句及其对应的单词词性和单词标签
# print(train_sents[0])  # 打印第0条训练语句及其对应的单词词性和单词标签


def word2featurevector(sent, i):  # 提取语句sent中的第i个单词的特征向量，并用一个string list：features表示
    # 19 attributes
    # w[t-2]                      w[t-1]        w[t]        w[t+1]                      w[t+2]
    #                             w[t-1]|w[t]          w[t]|w[t+1]
    # pos[t-2]                    pos[t-1]     pos[t]     pos[t+1]                    pos[t+2]
    # pos[t-2]|pos[t-1]           pos[t-1]|pos[t]  pos[t]|pos[t+1]           pos[t+1]|pos[t+2]
    # pos[t-2]|pos[t-1]|pos[t]        pos[t-1]|pos[t]|pos[t+1]        pos[t]|pos[t+1]|pos[t+2]
    sentence_length = len(sent)
    word_2 = sent[(i - 2) % sentence_length][0].lower()  # 语句sent中的第i-2个单词word_2
    word_1 = sent[(i - 1) % sentence_length][0].lower()  # 第i-1个单词word_1
    word = (sent[i][0]).lower()  # 语句sent中的第i个单词word
    word1 = sent[(i + 1) % sentence_length][0].lower()  # 语句sent中的第i+1个单词word1
    word2 = sent[(i + 2) % sentence_length][0].lower()  # 语句sent中的第i+2个单词word2
    postag_2 = sent[(i - 2) % sentence_length][1]  # 语句sent中的第i-2个单词的词性postag_2
    postag_1 = sent[(i - 1) % sentence_length][1]  # 语句sent中的第i-1个单词的词性postag_1
    postag = sent[i][1]  # 语句sent中的第i个单词的词性postag
    postag1 = sent[(i + 1) % sentence_length][1]  # 语句sent中的第i+1个单词的词性postag1
    postag2 = sent[(i + 2) % sentence_length][1]  # 语句sent中的第i+2个单词的词性postag2

    if i == 0:
        word_2 = 'start_word_2'
        postag_2 = 'start_pos_2'
        word_1 = 'start_word_1'
        postag_1 = 'start_pos_1'
    if i == len(sent) - 1:
        word1 = 'end_word_1'
        postag1 = 'end_pos_1'
        word2 = 'end_word_2'
        postag2 = 'end_pos_2'
    if i == 1:
        word_2 = 'start_word_1'
        postag_2 = 'start_pos_1'
    if i == len(sent) - 2:
        word2 = 'end_word_1'
        postag2 = 'end_pos_1'

    feature_vector = [word_2, word_1, word, word1, word2,
                      ''.join([word_1, '|', word]), ''.join([word, '|', word1]),
                      postag_2, postag_1, postag, postag1, postag2,
                      ''.join([postag_2, '|', postag_1]), ''.join([postag_1, '|', postag]),
                      ''.join([postag, '|', postag1]), ''.join([postag1, '|', postag2]),
                      ''.join([postag_2, '|', postag_1, '|', postag]),
                      ''.join([postag_1, '|', postag, '|', postag1]),
                      ''.join([postag, '|', postag1, '|', postag2])]
    return feature_vector  # 第i个单词的特征向量


def sent2featurevectorsequence(sent):  # 提取一个句子sent的特征向量序列
    return [word2featurevector(sent, i) for i in range(len(sent))]


def sent2labelsequence(sent):  # 收集一个句子sent的标签序列
    return [label for token, postag, label in sent]


def sent2tokensequence(sent):  # 收集一个句子sent的token序列
    return [token for token, postag, label in sent]


def write_feature_value_table(t, x, y, target_filename):
    with open(target_filename, 'w') as file:
        for sentence_index in range(0, len(t)):
            for token_index in range(0, len(t[sentence_index])):
                line_content = ' '.join([
                    t[sentence_index][token_index],
                    ' '.join(x[sentence_index][token_index]),
                    y[sentence_index][token_index]])
                file.write(line_content + '\n')
            file.write('\n')


tokentrain = [sent2tokensequence(s) for s in train_sents]  # 提取“训练”数据集的token序列集合
xtrain = [sent2featurevectorsequence(s) for s in train_sents]  # 提取“训练”数据集的特征向量序列集合
ytrain = [sent2labelsequence(s) for s in train_sents]  # 提取“训练”数据集的标签序列集合
write_feature_value_table(tokentrain, xtrain, ytrain, 'train_reformatted.txt')

# print(xtrain[0])  # 打印第0个“训练”语句的特征向量序列
# print(ytrain[0])  # 打印第0个“测试”语句的特征向量序列

tokentest = [sent2tokensequence(s) for s in test_sents]  # 提取“测试”数据集的token序列集合
xtest = [sent2featurevectorsequence(s) for s in test_sents]  # 提取“测试”数据集的特征向量序列集合
ytest = [sent2labelsequence(s) for s in test_sents]  # 提取“测试”数据集的标签序列集合
write_feature_value_table(tokentest, xtest,  ytest, 'test_reformatted.txt')
