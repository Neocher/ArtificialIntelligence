#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>

from numpy import *
import re
import operator
import feedparser


def load_data_set():
    """
    产生测试用的数据集
    :return posting_list:列表，每个元素也是列表，表示一篇文章，文章列表由单词组成
    :return class_vec:列表，每篇文章的类型
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    提取出一系列文章出现过的所有词汇
    :param data_set:列表，每个元素也是列表，表示一篇文章，文章列表由单词组成
    :return:列表，表示这些文章出现过的所有词汇，每个元素是一个词汇
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def bag_of_words_to_vec(vocab_list, input_set):
    """
    获得每篇文章的特征向量
    :param vocab_list:列表，表示这些文章出现过的所有词汇，每个元素是一个词汇
    :param input_set:列表，某篇文章出现过的词汇
    :return:列表，词汇表里的词汇在这篇文章出现过为1，否则为0
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_naive_bayes(train_matrix, train_category):
    """
    获得一篇文章被归为某类的概率及已知被归为某类某个特征出现的概率的对数
    :param train_matrix:列表，列表的每个元素是一个列表，词汇表里的词汇在这篇文章出现过则这个列表中元素为1，否则为0
    :param train_category:列表，元素值为每篇文章分类结果
    :return p0_vec:列表，每个元素为浮点数，是已知被归为第一类某个特征出现的概率的对数
    :return p1_vec:列表，每个元素为浮点数，是已知被归为第二类某个特征出现的概率的对数
    :return p_abusive:浮点数，被归为某类的概率
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = log(p1_num / p1_denom)
    p0_vec = log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive


def train_naive_bayes_comp(train_matrix, train_category):
    """
    获得一个数据被归为某类的概率及已知被归为某类某个特征出现的概率的对数
    :param train_matrix:列表，列表的每个元素是一个列表，该条数据有这个特征则这个列表中元素为1，否则为0
    :param train_category:列表，元素值为分类结果
    :return p_vec:列表，列表的每个元素是一个列表，每个元素为浮点数，是已知被归为某一类某个特征出现的概率的对数
    :return p_abusive:列表，每个元素为浮点数，被归为某类的概率
    """
    num_train_set = len(train_matrix)
    num_attribute = len(train_matrix[0])
    sum_of_attribute = zeros((1, 10))
    for i in range(0, num_train_set):
        sum_of_attribute[0][train_category[i]] += 1
    p_abusive = list(sum_of_attribute[0] / num_train_set)
    p_num = ones((10, num_attribute))
    for i in range(0, num_train_set):
        p_num[train_category[i]] += train_matrix[i]
        # print train_matrix[i]
    p_vec = zeros((10, num_attribute))
    for i in range(0, 10):
        p_vec[i] = log(p_num[i] / (sum_of_attribute[0][i] + num_attribute))
    p_vec = list(p_vec)
    return p_vec, p_abusive


def classify_naive_bayes_comp(vec_to_classify, p_vec, p_abusive):
    """
    将某篇文章进行分类
    :param vec_to_classify:列表，这篇文章的特征向量
    :param p_vec:列表，列表的每个元素是一个列表，每个元素为浮点数，是已知被归为某一类某个特征出现的概率的对数
    :param p_abusive:列表，每个元素为浮点数，被归为某类的概率
    :return:整数，归类的结果
    """
    l = []
    max = 0
    for i in range(0, len(p_abusive)):
        p = sum(vec_to_classify * p_vec[i]) + log(p_abusive[i])
        l.append(p)
        if (p > l[max]):
            max = i
    return max


def classify_naive_bayes(vec_to_classify, p0_vec, p1_vec, p_class1):
    """
    将某篇文章进行分类
    :param vec_to_classify:列表，这篇文章的特征向量
    :param p0_vec:列表，每个元素为浮点数，是已知被归为第一类某个特征出现的概率的对数
    :param p1_vec:列表，每个元素为浮点数，是已知被归为第二类某个特征出现的概率的对数
    :param p_class:浮点数，被归为某类的概率
    :return:整数，归类的结果
    """
    p1 = sum(vec_to_classify * p1_vec) + log(p_class1)
    p0 = sum(vec_to_classify * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def text_parse(big_string):
    """
    从字符串中提取处它的所有单词
    :param big_string:字符串
    :return:列表，所有的出现过的单词，可重复
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_text():
    """
    检测邮件是否为垃圾邮件
    """
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = train_naive_bayes(array(train_mat), array(train_classes))
    print p0_vec
    error_count = 0
    for doc_index in test_set:
        word_vec = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes(array(word_vec), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1
    print 'the error rate is: ', float(error_count) / len(test_set)


def sign(i):
    if int(i) > 100:
        return 1
    else:
        return 0


def digit_recognizer():
    """
    分类数字
    """
    train_mat = []
    train_classes = []
    fr = open('train.csv')
    fr.readline()
    for line in fr.readlines():
        line_array = line.strip().split(',')
        train_classes.append(int(line_array[0]))
        train_data = []
        for i in range(1, len(line_array)):
            train_data.append(sign(line_array[i]))
        train_mat.append(train_data)
    p_vec, p_spam = train_naive_bayes_comp(array(train_mat), array(train_classes))
    # print p_vec[0]
    fr = open('train.csv')
    fr.readline()
    fw = open('result.csv', 'w')
    # fw.write('ImageId,Label\n')
    j = 0
    error = 0
    for line in fr.readlines():
        j += 1
        line_array = line.strip().split(',')
        test_data = []
        for i in range(1, len(line_array)):
            test_data.append(sign(line_array[i]))
        result = classify_naive_bayes_comp(test_data, p_vec, p_spam)
        label = train_classes[j - 1]
        if label != result:
            error += 1
            fw.write(str(label) + ',' + str(result) + '\n')
    print error
        # fw.write(str(j) + ',' + str(classify_naive_bayes_comp(test_data, p_vec, p_spam)) + '\n')
        # print classify_naive_bayes_comp(test_data, p_vec, p_spam)
    fw.close()


def calc_most_freq(vocab_list, full_text):
    """
    获取文章中出现过次数最多的n个词
    :param vocab_list:列表，文章中出现过的所有词，不重复
    :param full_text:列表，文章中出现过的所有词，不过滤重复的
    :return:列表，出现次数最多的n个词
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:10]


def local_words(feed1, feed0):
    """
    检测广告是哪个地区的
    :param feed1:词典，来自第一个订阅源的信息
    :param feed2:词典，来自第二个订阅源的信息
    :return:vocab_list:列表，所有用来做特征的单词
    :return:p0_vec:列表，每个元素为浮点数，是已知被归为第一类某个特征出现的概率的对数
    :return:p1_vec:列表，每个元素为浮点数，是已知被归为第二类某个特征出现的概率的对数
    """
    stop_word_list = ['a ', 'able ', 'about ', 'above ', 'abroad ', 'according ', 'accordingly ', 'across ',
                      'actually ',
                      'adj ', 'after ', 'afterwards ', 'again ', 'against ', 'ago ', 'ahead ', "ain't ", 'all ',
                      'allow ',
                      'allows ', 'almost ', 'alone ', 'along ', 'alongside ', 'already ', 'also ', 'although ',
                      'always ',
                      'am ', 'amid ', 'amidst ', 'among ', 'amongst ', 'an ', 'and ', 'another ', 'any ', 'anybody ',
                      'anyhow ', 'anyone ', 'anything ', 'anyway ', 'anyways ', 'anywhere ', 'apart ', 'appear ',
                      'appreciate ', 'appropriate ', 'are ', "aren't ", 'around ', 'as ', "a's ", 'aside ', 'ask ',
                      'asking ', 'associated ', 'at ', 'available ', 'away ', 'awfully ', 'b ', 'back ', 'backward ',
                      'backwards ', 'be ', 'became ', 'because ', 'become ', 'becomes ', 'becoming ', 'been ',
                      'before ',
                      'beforehand ', 'begin ', 'behind ', 'being ', 'believe ', 'below ', 'beside ', 'besides ',
                      'best ',
                      'better ', 'between ', 'beyond ', 'both ', 'brief ', 'but ', 'by ', 'c ', 'came ', 'can ',
                      'cannot ',
                      'cant ', "can't ", 'caption ', 'cause ', 'causes ', 'certain ', 'certainly ', 'changes ',
                      'clearly ',
                      "c'mon ", 'co ', 'co. ', 'com ', 'come ', 'comes ', 'concerning ', 'consequently ', 'consider ',
                      'considering ', 'contain ', 'containing ', 'contains ', 'corresponding ', 'could ', "couldn't ",
                      'course ', "c's ", 'currently ', 'd ', 'dare ', "daren't ", 'definitely ', 'described ',
                      'despite ',
                      'did ', "didn't ", 'different ', 'directly ', 'do ', 'does ', "doesn't ", 'doing ', 'done ',
                      "don't ",
                      'down ', 'downwards ', 'during ', 'e ', 'each ', 'edu ', 'eg ', 'eight ', 'eighty ', 'either ',
                      'else ', 'elsewhere ', 'end ', 'ending ', 'enough ', 'entirely ', 'especially ', 'et ', 'etc ',
                      'even ', 'ever ', 'evermore ', 'every ', 'everybody ', 'everyone ', 'everything ', 'everywhere ',
                      'ex ', 'exactly ', 'example ', 'except ', 'f ', 'fairly ', 'far ', 'farther ', 'few ', 'fewer ',
                      'fifth ', 'first ', 'five ', 'followed ', 'following ', 'follows ', 'for ', 'forever ', 'former ',
                      'formerly ', 'forth ', 'forward ', 'found ', 'four ', 'from ', 'further ', 'furthermore ', 'g ',
                      'get ', 'gets ', 'getting ', 'given ', 'gives ', 'go ', 'goes ', 'going ', 'gone ', 'got ',
                      'gotten ',
                      'greetings ', 'h ', 'had ', "hadn't ", 'half ', 'happens ', 'hardly ', 'has ', "hasn't ", 'have ',
                      "haven't ", 'having ', 'he ', "he'd ", "he'll ", 'hello ', 'help ', 'hence ', 'her ', 'here ',
                      'hereafter ', 'hereby ', 'herein ', "here's ", 'hereupon ', 'hers ', 'herself ', "he's ", 'hi ',
                      'him ', 'himself ', 'his ', 'hither ', 'hopefully ', 'how ', 'howbeit ', 'however ', 'hundred ',
                      'i ',
                      "i'd ", 'ie ', 'if ', 'ignored ', "i'll ", "i'm ", 'immediate ', 'in ', 'inasmuch ', 'inc ',
                      'inc. ',
                      'indeed ', 'indicate ', 'indicated ', 'indicates ', 'inner ', 'inside ', 'insofar ', 'instead ',
                      'into ', 'inward ', 'is ', "isn't ", 'it ', "it'd ", "it'll ", 'its ', "it's ", 'itself ',
                      "i've ",
                      'j ', 'just ', 'k ', 'keep ', 'keeps ', 'kept ', 'know ', 'known ', 'knows ', 'l ', 'last ',
                      'lately ', 'later ', 'latter ', 'latterly ', 'least ', 'less ', 'lest ', 'let ', "let's ",
                      'like ',
                      'liked ', 'likely ', 'likewise ', 'little ', 'look ', 'looking ', 'looks ', 'low ', 'lower ',
                      'ltd ',
                      'm ', 'made ', 'mainly ', 'make ', 'makes ', 'many ', 'may ', 'maybe ', "mayn't ", 'me ', 'mean ',
                      'meantime ', 'meanwhile ', 'merely ', 'might ', "mightn't ", 'mine ', 'minus ', 'miss ', 'more ',
                      'moreover ', 'most ', 'mostly ', 'mr ', 'mrs ', 'much ', 'must ', "mustn't ", 'my ', 'myself ',
                      'n ',
                      'name ', 'namely ', 'nd ', 'near ', 'nearly ', 'necessary ', 'need ', "needn't ", 'needs ',
                      'neither ', 'never ', 'neverf ', 'neverless ', 'nevertheless ', 'new ', 'next ', 'nine ',
                      'ninety ',
                      'no ', 'nobody ', 'non ', 'none ', 'nonetheless ', 'noone ', 'no-one ', 'nor ', 'normally ',
                      'not ',
                      'nothing ', 'notwithstanding ', 'novel ', 'now ', 'nowhere ', 'o ', 'obviously ', 'of ', 'off ',
                      'often ', 'oh ', 'ok ', 'okay ', 'old ', 'on ', 'once ', 'one ', 'ones ', "one's ", 'only ',
                      'onto ',
                      'opposite ', 'or ', 'other ', 'others ', 'otherwise ', 'ought ', "oughtn't ", 'our ', 'ours ',
                      'ourselves ', 'out ', 'outside ', 'over ', 'overall ', 'own ', 'p ', 'particular ',
                      'particularly ',
                      'past ', 'per ', 'perhaps ', 'placed ', 'please ', 'plus ', 'possible ', 'presumably ',
                      'probably ',
                      'provided ', 'provides ', 'q ', 'que ', 'quite ', 'qv ', 'r ', 'rather ', 'rd ', 're ', 'really ',
                      'reasonably ', 'recent ', 'recently ', 'regarding ', 'regardless ', 'regards ', 'relatively ',
                      'respectively ', 'right ', 'round ', 's ', 'said ', 'same ', 'saw ', 'say ', 'saying ', 'says ',
                      'second ', 'secondly ', 'see ', 'seeing ', 'seem ', 'seemed ', 'seeming ', 'seems ', 'seen ',
                      'self ',
                      'selves ', 'sensible ', 'sent ', 'serious ', 'seriously ', 'seven ', 'several ', 'shall ',
                      "shan't ",
                      'she ', "she'd ", "she'll ", "she's ", 'should ', "shouldn't ", 'since ', 'six ', 'so ', 'some ',
                      'somebody ', 'someday ', 'somehow ', 'someone ', 'something ', 'sometime ', 'sometimes ',
                      'somewhat ',
                      'somewhere ', 'soon ', 'sorry ', 'specified ', 'specify ', 'specifying ', 'still ', 'sub ',
                      'such ',
                      'sup ', 'sure ', 't ', 'take ', 'taken ', 'taking ', 'tell ', 'tends ', 'th ', 'than ', 'thank ',
                      'thanks ', 'thanx ', 'that ', "that'll ", 'thats ', "that's ", "that've ", 'the ', 'their ',
                      'theirs ', 'them ', 'themselves ', 'then ', 'thence ', 'there ', 'thereafter ', 'thereby ',
                      "there'd ", 'therefore ', 'therein ', "there'll ", "there're ", 'theres ', "there's ",
                      'thereupon ',
                      "there've ", 'these ', 'they ', "they'd ", "they'll ", "they're ", "they've ", 'thing ',
                      'things ',
                      'think ', 'third ', 'thirty ', 'this ', 'thorough ', 'thoroughly ', 'those ', 'though ', 'three ',
                      'through ', 'throughout ', 'thru ', 'thus ', 'till ', 'to ', 'together ', 'too ', 'took ',
                      'toward ',
                      'towards ', 'tried ', 'tries ', 'truly ', 'try ', 'trying ', "t's ", 'twice ', 'two ', 'u ',
                      'un ',
                      'under ', 'underneath ', 'undoing ', 'unfortunately ', 'unless ', 'unlike ', 'unlikely ',
                      'until ',
                      'unto ', 'up ', 'upon ', 'upwards ', 'us ', 'use ', 'used ', 'useful ', 'uses ', 'using ',
                      'usually ',
                      'v ', 'value ', 'various ', 'versus ', 'very ', 'via ', 'viz ', 'vs ', 'w ', 'want ', 'wants ',
                      'was ', "wasn't ", 'way ', 'we ', "we'd ", 'welcome ', 'well ', "we'll ", 'went ', 'were ',
                      "we're ",
                      "weren't ", "we've ", 'what ', 'whatever ', "what'll ", "what's ", "what've ", 'when ', 'whence ',
                      'whenever ', 'where ', 'whereafter ', 'whereas ', 'whereby ', 'wherein ', "where's ",
                      'whereupon ',
                      'wherever ', 'whether ', 'which ', 'whichever ', 'while ', 'whilst ', 'whither ', 'who ',
                      "who'd ",
                      'whoever ', 'whole ', "who'll ", 'whom ', 'whomever ', "who's ", 'whose ', 'why ', 'will ',
                      'willing ', 'wish ', 'with ', 'within ', 'without ', 'wonder ', "won't ", 'would ', "wouldn't ",
                      'x ',
                      'y ', 'yes ', 'yet ', 'you ', "you'd ", "you'll ", 'your ', "you're ", 'yours ', 'yourself ',
                      'yourselves ', "you've ", 'z ', 'zero ']
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    top_10_words = calc_most_freq(vocab_list, full_text)
    for word in top_10_words:
        vocab_list.remove(word[0])
    for word in vocab_list:
        if word in stop_word_list:
            vocab_list.remove(word)
    training_set = range(2 * min_len)
    test_set = []
    for i in range(15):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = train_naive_bayes(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vec = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes(array(word_vec), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1
    print 'the error rate is: ', float(error_count) / len(test_set)
    return vocab_list, p0_vec, p1_vec


def main():
    '''
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    local_words(ny, sf)
    '''
    digit_recognizer()


if __name__ == '__main__':
    main()